"""Deterministic institutional report (no API key required).

Assembles all of Agent 1's outputs into a Telegram-friendly plain-text report covering the
five requested deliverables plus the market snapshot, option context, and confidence
summary. An optional LLM narration layer (report/narrator.py) can enhance this later, but
this function alone produces the complete report.
"""
from __future__ import annotations

from src.features.options import OptionFeatures
from src.trade.engine import TradePlan


def _bar(pct: float, width: int = 10) -> str:
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)


def build_report(
    *, date_str: str, mode: str, open_price: float, prev_close: float,
    gap_pct: float, india_vix: float | None,
    or_ret: float | None, or_range: float | None,
    pred_low: float, pred_high: float, expected_move_pts: float,
    p_up: float, exp_magnitude_pts: float, bias: str,
    p_down: float, p_sideways: float, p_up_reg: float,
    conf_direction: float, conf_regime: float, conf_range: float, conf_overall: float,
    plans: list[TradePlan], opt: OptionFeatures | None, global_summary: str,
) -> str:
    L: list[str] = []
    L.append(f"📊 NIFTY 50 — Pre-Market Outlook ({mode})")
    L.append(f"🗓️ {date_str}")
    L.append("")

    # Snapshot
    L.append("— MARKET SNAPSHOT —")
    L.append(f"Open: {open_price:,.0f}  | Prev close: {prev_close:,.0f}  | Gap: {gap_pct:+.2%}")
    if india_vix is not None:
        L.append(f"India VIX: {india_vix:.2f}")
    if or_ret is not None:
        L.append(f"Opening range (9:15-9:20): move {or_ret:+.2%}, range {or_range:.2%}")
    L.append(f"Global: {global_summary}")
    if opt:
        L.append(f"Options (exp {opt.expiry}): PCR {opt.pcr} | MaxPain {int(opt.max_pain)} "
                 f"| Put wall {int(opt.put_wall)} | Call wall {int(opt.call_wall)} "
                 f"| ATM IV {opt.atm_iv}% | Skew {opt.iv_skew:+.1f}")
    L.append("")

    # 1. Expected Day Range
    L.append("1️⃣ EXPECTED DAY RANGE")
    L.append(f"   {pred_low:,.0f}  —  {pred_high:,.0f}   (±{expected_move_pts:,.0f} pts)")
    L.append(f"   confidence {conf_range:.0f}/100  {_bar(conf_range)}")
    L.append("")

    # 2. Close vs Open Direction (binary model: which side of the open)
    L.append("2️⃣ CLOSE vs OPEN DIRECTION")
    L.append(f"   Bias: {bias}  |  P(up) {p_up:.0%} / P(down) {1 - p_up:.0%}")
    L.append(f"   Expected close-open: {exp_magnitude_pts:+,.0f} pts")
    L.append(f"   confidence {conf_direction:.0f}/100  {_bar(conf_direction)}")
    L.append("")

    # 3. Probability Model — move *size* regime (decisive move vs flat)
    L.append("3️⃣ PROBABILITY MODEL (move size)")
    L.append(f"   Decisive Up {p_up_reg:.0%}  | Decisive Down {p_down:.0%}  | Flat/Range {p_sideways:.0%}")
    L.append("   (decisive = |move| > 0.25%; the direction call above integrates these)")
    # Flag when the two models disagree on sign — treat the call cautiously.
    if (p_up >= 0.5) != (p_up_reg >= p_down):
        L.append("   ⚠ direction & regime models diverge on sign — lower conviction")
    L.append(f"   confidence {conf_regime:.0f}/100  {_bar(conf_regime)}")
    L.append("")

    # 4. Trade Plans
    L.append("4️⃣ TRADER-SPECIFIC PLANS")
    for pl in plans:
        flag = "✅ TAKE" if pl.take_trade else "⏸️ SKIP"
        L.append(f"   • {pl.persona}  [{flag}, conf {pl.confidence:.0f}]")
        L.append(f"     {pl.summary}")
        if pl.stop_loss:
            L.append(f"     SL: {pl.stop_loss}")
        if pl.target:
            L.append(f"     Target: {pl.target}")
        if pl.rr:
            L.append(f"     RR: {pl.rr}")
        L.append(f"     ↳ {pl.rationale}")
    L.append("")

    # 5. Confidence summary
    L.append("5️⃣ CONFIDENCE")
    L.append(f"   Range {conf_range:.0f} | Direction {conf_direction:.0f} | "
             f"Probability {conf_regime:.0f}")
    L.append(f"   OVERALL {conf_overall:.0f}/100  {_bar(conf_overall)}")
    L.append("")
    L.append("⚠️ Probabilistic model output, not financial advice. "
             "Manage risk; size positions yourself.")
    return "\n".join(L)
