# Crypto Portfolio Dashboard

Streamlit dashboard that parses your existing `Crypto Portfolio.xlsx` and augments with live prices (CoinGecko) for real‑time metrics, allocation, and P&L.

## Features
* Parses messy header region to extract transaction rows
* Aggregates current positions
* Live price lookup (GBP/USD/EUR) via CoinGecko public API
* Allocation pie, unrealized P&L bar, cumulative invested capital timeline
* Auto refresh & manual refresh
* .env driven configuration

## Quick Start

```bash
# Activate virtual environment if not yet active
source .venv/bin/activate  # (macOS/Linux)

# Copy environment template
cp .env.example .env
# (Optionally set COINGECKO_API_KEY if you have Pro)

# Run dashboard
streamlit run dashboard_app.py
```

Open the displayed local URL in a browser.

## Notes
* The Excel structure was inferred; adjust parsing logic in `portfolio_loader.py` if needed.
* Live prices use CoinGecko free endpoints (rate-limited). For many symbols you may need to slow refresh period.
* Verify financial figures before making decisions.

## Next Ideas
* Persist historical snapshots
* Add staking/yield tracking
* Support multiple wallets & chains via API keys (e.g., Etherscan)
* Authentication
