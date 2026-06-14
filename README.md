# Nifty Intraday Prediction — Two-Agent Self-Improving System

An institutional-style, **self-improving** system that predicts Nifty 50 intraday behaviour
each morning and reviews + improves itself each evening, delivering to Telegram.

> ⚠️ Probabilistic research tool, **not financial advice**. Markets are largely efficient;
> the edge here is modest and risk-managed, not a crystal ball. Trade your own risk.

## The two agents

**Agent 1 — Pre-Market Predictor** (~09:20 IST)
Live Dhan data (opening-range candle + option chain) → engineered features → calibrated
champion model → 5 outputs delivered to Telegram:
1. Expected day range
2. Close-vs-open direction
3. Up / Down / Sideways probabilities
4. Trade plans for 4 personas (non-directional seller, directional seller, option buyer, futures)
5. Confidence per component + overall

**Agent 2 — Reviewer / Auto-Healer** (~20:00 IST)
Pulls the actual outcome, scores the morning call (direction, range hit, Brier calibration,
trade P&L, confidence reliability), judges "satisfactory" against a rolling threshold, and
runs a **champion/challenger** loop that only promotes a new model if it beats the current
one on backtest. Sends an evening review.

## Methodology (beyond simple indicators)

- Range-based volatility: Parkinson, Garman-Klass, **GKYZ**, **Yang-Zhang**
- Asymmetric **EGARCH + TGARCH** conditional volatility; India-VIX-implied move
- Direction/regime: **XGBoost + LightGBM + logistic** ensemble, isotonic-calibrated
- Markov regime, gap, VIX bands, momentum, and the **09:15–09:20 opening-range** as features
- Honest, walk-forward backtested; self-improvement gated on backtest performance

## Layout

```
src/data/        Dhan client, yfinance history, intraday/opening-range backfill, global cues
src/features/    volatility, garch, regime, technical, options, builder
src/models/      direction, regime, range, confidence, predictor, registry, train
src/backtest/    walk-forward harness
src/scoring/     scorecard + outcome review
src/improve/     judge, diagnose, champion/challenger
src/trade/       4-persona trade engine
src/report/      deterministic report builder
src/delivery/    Telegram
scripts/         train_initial, run_agent1, run_agent2
config/          settings.yaml (base) + learned.yaml (Agent 2-tuned)
data/            historical cache, logs (predictions/outcomes/metrics), model registry
```

## Setup

```bash
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
copy .env.example .env       # fill in tokens (NEVER commit .env)
python -m scripts.train_initial   # build dataset + train champion (uses Dhan opening-range cache)
python -m scripts.run_agent1 --dry-run
python -m scripts.run_agent2 --dry-run
```

### Required secrets (`.env` locally, **GitHub Actions secrets** for automation)

| Secret | Purpose |
|---|---|
| `DHAN_ACCESS_TOKEN` | Dhan API (expires daily — refresh each morning during testing) |
| `DHAN_CLIENT_ID` | Dhan client id |
| `TELEGRAM_BOT_TOKEN` | Telegram delivery |
| `TELEGRAM_CHAT_ID` | Telegram chat id |

`ANTHROPIC_API_KEY` is optional (richer narration); the system runs fully without it.

## Automation

GitHub Actions run Agent 1 (`.github/workflows/agent1.yml`, ~09:20 IST) and Agent 2
(`.github/workflows/agent2.yml`, ~20:00 IST) on weekdays, committing logs/models back to
the repo. Add the four secrets above under **Settings → Secrets and variables → Actions**.
During the testing phase, update `DHAN_ACCESS_TOKEN` each morning (it expires daily); a VPS
with automated token refresh is the planned next step.
