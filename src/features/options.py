"""Option-chain feature extraction from a Dhan option-chain response.

These are LIVE-only signals (no deep history), used by the trade engine and as a
confidence overlay — not by the trained core models. They encode where the option
market sees support/resistance and how it is pricing risk:

    spot, atm_strike
    pcr                 total PUT OI / total CALL OI   (>1 = put-heavy / supportive)
    max_pain            strike where option writers lose least (pin magnet)
    call_wall           strike with the most CALL OI    -> resistance
    put_wall            strike with the most PUT OI      -> support
    atm_iv              average ATM implied vol (%)
    iv_skew             OTM put IV - OTM call IV (%)      (>0 = downside fear)
    atm_ce_ltp/atm_pe_ltp  ATM premiums (for trade sizing)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptionFeatures:
    spot: float
    atm_strike: float
    pcr: float
    max_pain: float
    call_wall: float
    put_wall: float
    atm_iv: float
    iv_skew: float
    atm_ce_ltp: float
    atm_pe_ltp: float
    expiry: str
    strike_step: float
    ce_ltp: dict[float, float]   # strike -> call last price
    pe_ltp: dict[float, float]   # strike -> put last price

    def _nearest(self, mapping: dict[float, float], strike: float) -> float | None:
        if not mapping:
            return None
        if strike in mapping:
            return mapping[strike]
        k = min(mapping.keys(), key=lambda s: abs(s - strike))
        return mapping[k]

    def call_premium(self, strike: float) -> float | None:
        return self._nearest(self.ce_ltp, strike)

    def put_premium(self, strike: float) -> float | None:
        return self._nearest(self.pe_ltp, strike)


def _normalize(oc_response: dict) -> tuple[float, dict]:
    """Return (spot, {strike_float: {'ce':..., 'pe':...}}) from a Dhan response."""
    data = oc_response.get("data", oc_response)
    spot = float(data.get("last_price"))
    raw = data.get("oc", {})
    chain = {}
    for k, v in raw.items():
        try:
            chain[float(k)] = v
        except (TypeError, ValueError):
            continue
    return spot, chain


def _oi(leg: dict) -> float:
    try:
        return float(leg.get("oi") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _iv(leg: dict) -> float:
    try:
        return float(leg.get("implied_volatility") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _ltp(leg: dict) -> float:
    try:
        return float(leg.get("last_price") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _max_pain(chain: dict) -> float:
    """Strike minimizing total intrinsic payout to option buyers at expiry."""
    strikes = sorted(chain.keys())
    best_k, best_loss = strikes[0], float("inf")
    for settle in strikes:
        loss = 0.0
        for k in strikes:
            ce_oi = _oi(chain[k].get("ce", {}))
            pe_oi = _oi(chain[k].get("pe", {}))
            loss += ce_oi * max(0.0, settle - k)   # calls ITM if settle > strike
            loss += pe_oi * max(0.0, k - settle)   # puts ITM if settle < strike
        if loss < best_loss:
            best_loss, best_k = loss, settle
    return best_k


def extract_option_features(oc_response: dict, expiry: str) -> OptionFeatures:
    spot, chain = _normalize(oc_response)
    strikes = sorted(chain.keys())
    atm = min(strikes, key=lambda s: abs(s - spot))
    step = min((b - a) for a, b in zip(strikes[:-1], strikes[1:]))

    total_ce_oi = sum(_oi(chain[k].get("ce", {})) for k in strikes)
    total_pe_oi = sum(_oi(chain[k].get("pe", {})) for k in strikes)
    pcr = (total_pe_oi / total_ce_oi) if total_ce_oi else 0.0

    call_wall = max(strikes, key=lambda k: _oi(chain[k].get("ce", {})))
    put_wall = max(strikes, key=lambda k: _oi(chain[k].get("pe", {})))

    atm_ce = chain[atm].get("ce", {})
    atm_pe = chain[atm].get("pe", {})
    atm_iv = (_iv(atm_ce) + _iv(atm_pe)) / 2.0

    # IV skew: compare IVs roughly 2% OTM on each side.
    otm_put_k = min(strikes, key=lambda s: abs(s - spot * 0.98))
    otm_call_k = min(strikes, key=lambda s: abs(s - spot * 1.02))
    iv_skew = _iv(chain[otm_put_k].get("pe", {})) - _iv(chain[otm_call_k].get("ce", {}))

    ce_ltp = {k: _ltp(chain[k].get("ce", {})) for k in strikes}
    pe_ltp = {k: _ltp(chain[k].get("pe", {})) for k in strikes}

    return OptionFeatures(
        spot=spot, atm_strike=atm, pcr=round(pcr, 3), max_pain=_max_pain(chain),
        call_wall=call_wall, put_wall=put_wall, atm_iv=round(atm_iv, 2),
        iv_skew=round(iv_skew, 2),
        atm_ce_ltp=_ltp(atm_ce), atm_pe_ltp=_ltp(atm_pe),
        expiry=expiry, strike_step=step, ce_ltp=ce_ltp, pe_ltp=pe_ltp,
    )
