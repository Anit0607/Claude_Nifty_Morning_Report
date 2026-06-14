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

    # ---- 1. Non-Directional Seller (short strangle) ----
    width = p["non_directional_seller"]["strike_width_atm_mult"] * move_pts
    call_k = _round_to(open_price + width, step)
    put_k = _round_to(open_price - width, step)
    if opt:  # widen to the OI walls if they sit further out (safer shorts)
        call_k = max(call_k, opt.call_wall)
        put_k = min(put_k, opt.put_wall)
    iv_ok = (opt.atm_iv if opt else 99) >= p["non_directional_seller"]["min_vix"]
    nds_conf = round(0.5 * conf_range + 0.5 * min(99.0, 50 + 100 * (p_sideways - 1 / 3)), 1)
    plans.append(TradePlan(
        persona="Intraday Option Non-Directional Seller",
        bias="Neutral / Range",
        take_trade=bool(iv_ok and p_sideways >= max(p_down, p_up_reg) * 0.9),
        confidence=nds_conf,
        summary=f"Short strangle: sell {int(call_k)}CE + sell {int(put_k)}PE",
        legs=[f"SELL {int(call_k)} CE", f"SELL {int(put_k)} PE"],
        stop_loss="Exit a side if Nifty trades through that short strike, or if combined "
                  "premium rises ~1.7x collected.",
        target="~50% of premium collected (time decay).",
        rr="Theta-positive; defined-risk if hedged into an iron condor.",
        rationale=f"Expected move ±{move_pts:,.0f} pts; sideways prob {p_sideways:.0%}. "
                  f"{'IV adequate' if iv_ok else 'IV low — premium thin, caution'}."
                  + (f" Anchored to OI walls {int(opt.put_wall)}/{int(opt.call_wall)}." if opt else ""),
    ))

    # ---- 2. Directional Seller (sell the OTM opposite side) ----
    # Sell at the buffer strike for meaningful premium; only snap to an OI wall when that
    # wall sits *near* the buffer strike (so we align to real support/resistance without
    # chasing a far-OTM wall that collects almost no premium).
    buf = p["directional_seller"]["otm_buffer_mult"] * move_pts
    snap = 0.5 * move_pts  # how close a wall must be to override the buffer strike
    if bullish:
        sell_k = _round_to(open_price - buf, step)
        if opt and open_price >= opt.put_wall >= sell_k - snap:
            sell_k = opt.put_wall   # strong support just below our strike -> sell there
        ds_summary = f"Sell {int(sell_k)} PE (bullish: collect premium below support)"
        ds_legs = [f"SELL {int(sell_k)} PE"]
    else:
        sell_k = _round_to(open_price + buf, step)
        if opt and open_price <= opt.call_wall <= sell_k + snap:
            sell_k = opt.call_wall  # strong resistance just above our strike -> sell there
        ds_summary = f"Sell {int(sell_k)} CE (bearish: collect premium above resistance)"
        ds_legs = [f"SELL {int(sell_k)} CE"]
    plans.append(TradePlan(
        persona="Intraday Option Directional Seller",
        bias=bias,
        take_trade=bool(conviction >= 0.12),
        confidence=round(conf_direction, 1),
        summary=ds_summary,
        legs=ds_legs,
        stop_loss="Exit if Nifty trades through the short strike or premium ~2x.",
        target="~50% premium decay or close of session.",
        rr="Theta-positive; directional risk if trend accelerates against the short.",
        rationale=f"Model bias {bias.lower()} (P_up={p_up:.0%}, conviction {conviction:.0%}); "
                  f"sells the side the move is least likely to reach.",
    ))

    # ---- 3. Option Buyer (gated to high conviction) ----
    min_prob = p["option_buyer"]["min_directional_prob"]
    dir_prob = p_up if bullish else (1 - p_up)
    buy_k = _round_to(open_price, step)  # ATM
    side = "CE" if bullish else "PE"
    tgt_pts = max(move_pts, 1.5 * atr_points)
    plans.append(TradePlan(
        persona="Intraday Option Buyer",
        bias=bias,
        take_trade=bool(dir_prob >= min_prob),
        confidence=round(conf_direction * (1.0 if dir_prob >= min_prob else 0.6), 1),
        summary=f"Buy {int(buy_k)} {side} (ATM) — directional",
        legs=[f"BUY {int(buy_k)} {side}"],
        stop_loss="~35-40% of premium paid (or underlying back through the open).",
        target=f"Underlying move ~{tgt_pts:,.0f} pts toward "
               f"{int(pred_high) if bullish else int(pred_low)} (designed ~1:2 on premium).",
        rr="~1:2 by design; only taken when directional probability clears the gate.",
        rationale=f"Directional prob {dir_prob:.0%} vs gate {min_prob:.0%}: "
                  f"{'TAKE' if dir_prob >= min_prob else 'SKIP — conviction too low'}.",
    ))

    # ---- 4. Futures Trader (ATR bracket defines the RR) ----
    ft = p["futures_trader"]
    sl_pts = ft["sl_atr_mult"] * atr_points
    tp_pts = ft["target_atr_mult"] * atr_points
    rr = tp_pts / sl_pts if sl_pts else 0.0
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
        stop_loss=f"{int(ft_sl)} ({sl_pts:,.0f} pts, {ft['sl_atr_mult']}xATR)",
        target=f"{int(ft_tp)} ({tp_pts:,.0f} pts, {ft['target_atr_mult']}xATR)",
        rr=f"~1:{rr:.1f}",
        rationale=f"Trade the model bias with an ATR bracket; "
                  f"{'TAKE' if conviction >= 0.08 else 'stand aside (no conviction)'}.",
    ))

    return plans
