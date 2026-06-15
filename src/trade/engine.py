"""Trade-plan engine: model outputs + option chain -> 4 persona plans.

Personas:
  1. Non-Directional Seller  — short strangle, profits from time decay in a range
  2. Directional Seller      — sells the far OTM side opposite the predicted move
  3. Option Buyer            — buys direction; gated to high-conviction days only
  4. Futures Trader          — intraday futures with ATR-based bracket (defines the RR)

Design choices reflecting the validated edge:
  * Direction is a modest edge, so the **Option Buyer and Futures** trades are gated by a
    confidence/probability threshold — we skip coin-flip days rather than force a trade.
  * Volatility/range is the more reliable output, so the **seller** structures lean on the
    expected range and the option OI walls (natural support/resistance).

Each plan carries its own confidence and a ``take_trade`` flag (the selectivity gate).
Levels are in index points; sellers reference option strikes snapped to the chain step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import load_settings
from src.features.options import OptionFeatures


@dataclass
class TradePlan:
    persona: str
    bias: str
    take_trade: bool
    confidence: float
    summary: str
    legs: list[str] = field(default_factory=list)
    stop_loss: str = ""
    target: str = ""
    rr: str = ""
    rationale: str = ""


def _round_to(x: float, step: float) -> float:
    return round(x / step) * step


def build_trade_plans(
    *, open_price: float, expected_move: float, pred_high: float, pred_low: float,
    p_up: float, p_down: float, p_sideways: float, p_up_reg: float,
    conf_direction: float, conf_regime: float, conf_range: float,
    atr_points: float, opt: OptionFeatures | None,
) -> list[TradePlan]:
    s = load_settings()["personas"]
    p = s
    move_pts = open_price * expected_move
    step = opt.strike_step if opt else 50.0
    conviction = abs(p_up - 0.5) * 2.0            # 0 (coin flip) .. 1 (certain)
    bullish = p_up >= 0.5
    bias = "Bullish" if bullish else "Bearish"

    plans: list[TradePlan] = []

    # ---- 1. Non-Directional Seller (short strangle), premium-based 1:1 ----
    nds = p["non_directional_seller"]
    width = nds["strike_width_atm_mult"] * move_pts
    call_k = _round_to(open_price + width, step)
    put_k = _round_to(open_price - width, step)
    if opt:  # widen to the OI walls if they sit further out (safer shorts)
        call_k = max(call_k, opt.call_wall)
        put_k = min(put_k, opt.put_wall)
    iv_ok = (opt.atm_iv if opt else 99) >= nds["min_vix"]
    nds_conf = round(0.5 * conf_range + 0.5 * min(99.0, 50 + 100 * (p_sideways - 1 / 3)), 1)
    combined = None
    if opt:
        cp, pp = opt.call_premium(call_k), opt.put_premium(put_k)
        if cp and pp:
            combined = cp + pp
    sl_line, tgt_line = _seller_risk(combined, nds["sl_premium_pct"],
                                     nds["target_premium_pct"], nds["trail"], per_side=True)
    plans.append(TradePlan(
        persona="Intraday Option Non-Directional Seller",
        bias="Neutral / Range",
        take_trade=bool(iv_ok and p_sideways >= max(p_down, p_up_reg) * 0.9),
        confidence=nds_conf,
        summary=f"Short strangle: sell {int(call_k)}CE + sell {int(put_k)}PE"
                + (f" (premium ~{combined:.1f})" if combined else ""),
        legs=[f"SELL {int(call_k)} CE", f"SELL {int(put_k)} PE"],
        stop_loss=sl_line, target=tgt_line,
        rr=f"1:1 on premium ({nds['target_premium_pct']:.0%} decay vs {nds['sl_premium_pct']:.0%} stop)",
        rationale=f"Expected move ±{move_pts:,.0f} pts; sideways prob {p_sideways:.0%}. "
                  f"{'IV adequate' if iv_ok else 'IV low — premium thin, caution'}."
                  + (f" Anchored to OI walls {int(opt.put_wall)}/{int(opt.call_wall)}." if opt else ""),
    ))

    # ---- 2. Directional Seller (sell the OTM opposite side), premium-based 1:1 ----
    ds = p["directional_seller"]
    buf = ds["otm_buffer_mult"] * move_pts
    snap = 0.5 * move_pts  # how close a wall must be to override the buffer strike
    if bullish:
        sell_k = _round_to(open_price - buf, step)
        if opt and open_price >= opt.put_wall >= sell_k - snap:
            sell_k = opt.put_wall
        prem = opt.put_premium(sell_k) if opt else None
        ds_summary = f"Sell {int(sell_k)} PE (bullish: collect premium below support)"
        ds_legs = [f"SELL {int(sell_k)} PE"]
    else:
        sell_k = _round_to(open_price + buf, step)
        if opt and open_price <= opt.call_wall <= sell_k + snap:
            sell_k = opt.call_wall
        prem = opt.call_premium(sell_k) if opt else None
        ds_summary = f"Sell {int(sell_k)} CE (bearish: collect premium above resistance)"
        ds_legs = [f"SELL {int(sell_k)} CE"]
    if prem:
        ds_summary += f" @ ~{prem:.1f}"
    sl_line, tgt_line = _seller_risk(prem, ds["sl_premium_pct"], ds["target_premium_pct"], ds["trail"])
    plans.append(TradePlan(
        persona="Intraday Option Directional Seller",
        bias=bias,
        take_trade=bool(conviction >= 0.12),
        confidence=round(conf_direction, 1),
        summary=ds_summary, legs=ds_legs, stop_loss=sl_line, target=tgt_line,
        rr=f"1:1 on premium ({ds['target_premium_pct']:.0%} decay vs {ds['sl_premium_pct']:.0%} stop)",
        rationale=f"Model bias {bias.lower()} (P_up={p_up:.0%}, conviction {conviction:.0%}); "
                  f"premium-based stop beats strike-breach exit on POP.",
    ))

    # ---- 3. Option Buyer (gated to high conviction) ----
    ob = p["option_buyer"]
    min_prob = ob["min_directional_prob"]
    dir_prob = p_up if bullish else (1 - p_up)
    buy_k = _round_to(open_price, step)  # ATM
    side = "CE" if bullish else "PE"
    prem = (opt.call_premium(buy_k) if bullish else opt.put_premium(buy_k)) if opt else None
    if prem:
        sl_prem = prem * (1 - ob["sl_premium_pct"])
        tgt_prem = prem * (1 + ob["target_rr"] * ob["sl_premium_pct"])
        ob_sl = f"premium {prem:.1f} → SL {sl_prem:.1f} (−{ob['sl_premium_pct']:.0%})"
        ob_tgt = f"target {tgt_prem:.1f} (+{ob['target_rr']*ob['sl_premium_pct']:.0%}, ~1:{ob['target_rr']:.0f})"
    else:
        ob_sl = f"SL −{ob['sl_premium_pct']:.0%} of premium paid"
        ob_tgt = f"target ~1:{ob['target_rr']:.0f} on premium"
    plans.append(TradePlan(
        persona="Intraday Option Buyer",
        bias=bias,
        take_trade=bool(dir_prob >= min_prob),
        confidence=round(conf_direction * (1.0 if dir_prob >= min_prob else 0.6), 1),
        summary=f"Buy {int(buy_k)} {side} (ATM)" + (f" @ ~{prem:.1f}" if prem else "") + " — directional",
        legs=[f"BUY {int(buy_k)} {side}"],
        stop_loss=ob_sl, target=ob_tgt, rr=f"~1:{ob['target_rr']:.0f}",
        rationale=f"Directional prob {dir_prob:.0%} vs gate {min_prob:.0%}: "
                  f"{'TAKE' if dir_prob >= min_prob else 'SKIP — conviction too low'}.",
    ))

    # ---- 4. Futures Trader: hard points SL + trailing (user's risk style) ----
    ft = p["futures_trader"]
    sl_pts = float(ft["sl_points"])
    tp_pts = ft["target_rr"] * sl_pts
    trail = float(ft["trail_step"])
    if bullish:
        ft_sl, ft_tp = open_price - sl_pts, open_price + tp_pts
    else:
        ft_sl, ft_tp = open_price + sl_pts, open_price - tp_pts
    plans.append(TradePlan(
        persona="Intraday Futures Trader",
        bias=bias,
        take_trade=bool(conviction >= 0.08),
        confidence=round(conf_direction, 1),
        summary=f"{'Long' if bullish else 'Short'} Nifty fut from ~{int(open_price)}",
        legs=[f"{'BUY' if bullish else 'SELL'} NIFTY FUT @ {int(open_price)}"],
        stop_loss=f"{int(ft_sl)} (hard {sl_pts:.0f} pts)",
        target=f"first {int(ft_tp)} (1:{ft['target_rr']:.0f}), then TRAIL SL by {trail:.0f} pts "
               f"per {trail:.0f} pts in favour (breakeven at 1:1, let winner run)",
        rr=f"1:{ft['target_rr']:.0f} then trailing",
        rationale=f"Trade the model bias; tight {sl_pts:.0f}-pt stop, trail to lock profit. "
                  f"{'TAKE' if conviction >= 0.08 else 'stand aside (no conviction)'}.",
    ))

    return plans


def _seller_risk(premium: float | None, sl_pct: float, tgt_pct: float,
                 trail: bool, per_side: bool = False) -> tuple[str, str]:
    """Build premium-based stop/target lines for an option-selling plan."""
    scope = "combined premium" if per_side else "premium"
    if premium:
        sl = premium * (1 + sl_pct)
        tgt = premium * (1 - tgt_pct)
        sl_line = f"{scope} {premium:.1f} → SL {sl:.1f} (+{sl_pct:.0%})"
        tgt_line = f"take profit {tgt:.1f} (−{tgt_pct:.0%}, 1:1)"
    else:
        sl_line = f"SL when {scope} rises +{sl_pct:.0%}"
        tgt_line = f"take profit at −{tgt_pct:.0%} {scope} decay (1:1)"
    if trail:
        tgt_line += "; trail stop down as premium decays to lock profit"
    return sl_line, tgt_line
