# EDGAR Financial Dashboard

Pulls all data live from SEC EDGAR.
No API keys beyond a free SEC identity email.

---

## Quick Start

### 1. Clone / download

Put `edgar_dashboard.py`, `requirements.txt`, and the `.streamlit/` folder
in the same directory.

```
my-dashboard/
├── edgar_dashboard.py
├── requirements.txt
└── .streamlit/
    └── config.toml
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Mac / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
streamlit run edgar_dashboard.py
```

Your browser opens automatically at `http://localhost:8501`.

---

## Usage

1. Enter a **ticker symbol** in the left sidebar (e.g. `AAPL`, `MSFT`, `DPZ`).
2. Enter any **email address** — SEC EDGAR requires an identity string per their
   fair-use policy. It is never stored anywhere; it is only passed as a
   request header to `efts.sec.gov`.
3. Click **Fetch Data**.

Data is cached for **1 hour** per ticker. Subsequent renders are instant.
Click Fetch Data again to force a refresh.

---

## Tabs

| Tab | What you get |
|-----|-------------|
| 📌 Valuation | Live multiples (P/E, P/S, P/FCF, P/CFO, P/GP, P/Book, EV/EBITDA, EV/Sales), EPS, FCF/share |
| 📈 Income | Annual revenue, gross profit, operating income, net income — bars + margin lines |
| 🏦 Balance Sheet | Assets / liabilities / equity, debt vs cash over time |
| 💵 Cash Flow | OCF, CapEx, FCF — plus FCF vs Net Income (Sloan accrual signal) |
| 📅 Quarterly | Single-quarter breakdowns: revenue, margins, cash flow |
| 🔄 Consistency | ConsistencyScore — #Rec (all-time highs) + #Inc (YoY increases) per metric, per-share series explorer |
| 📊 CAGRs | 1–14 year CAGR table + heatmap, per-share basis |
| 🗂 Raw XBRL | Every raw EDGAR concept, searchable, flow vs balance-sheet split |

---

## Requirements

| Package | Why |
|---------|-----|
| `streamlit` | UI framework |
| `edgartools` | Pulls XBRL facts from SEC EDGAR |
| `yfinance` | Current price and market cap |
| `pandas` | Data manipulation |
| `numpy` | Numerics |
| `plotly` | Charts |

---

## Troubleshooting

**"No EDGAR facts for XYZ"**  
The ticker may not file with the SEC (foreign private issuer, OTC, or
delisted). Try the primary exchange ticker.

**Slow first load**  
First fetch hits EDGAR and yfinance live — usually 5–15 seconds.
All subsequent interactions within the 1-hour cache window are instant.

**"ModuleNotFoundError: edgartools"**  
Run `pip install edgartools` — the package name on PyPI is `edgartools`,
not `edgar`.

**Port already in use**  
Run `streamlit run edgar_dashboard.py --server.port 8502` to use a
different port.
