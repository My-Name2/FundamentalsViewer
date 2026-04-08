"""
edgar_dashboard.py  —  SEC EDGAR Financial Dashboard (Streamlit)

Install:
    pip install streamlit edgartools yfinance pandas numpy plotly

Run:
    streamlit run edgar_dashboard.py
"""

import math
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDGAR Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Minimal dark-ish style patch (works in both themes)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e2433;
        border: 1px solid #2e3550;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .metric-card .label { font-size: 11px; color: #8892a4; text-transform: uppercase; letter-spacing: .5px; }
    .metric-card .value { font-size: 22px; font-weight: 700; color: #e8ecf4; margin-top: 2px; }
    .flag-high   { color: #ff6b6b; font-weight: 700; }
    .flag-mod    { color: #ffb347; font-weight: 700; }
    .flag-clean  { color: #6bcb77; font-weight: 700; }
    .section-header {
        font-size: 15px; font-weight: 600; color: #8ab4f8;
        border-bottom: 1px solid #2e3550; padding-bottom: 4px; margin: 18px 0 10px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _safe(val):
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except Exception:
        return None


def fmt_val(v, unit="$B"):
    """Format a raw-dollar value (not in millions) for display."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    try:
        v = float(v)
    except Exception:
        return "—"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if unit == "$B":
        if v >= 1e12: return f"{sign}${v/1e12:.2f}T"
        if v >= 1e9:  return f"{sign}${v/1e9:.2f}B"
        if v >= 1e6:  return f"{sign}${v/1e6:.1f}M"
        return f"{sign}${v:,.0f}"
    return f"{sign}{v:.2f}"


def fmt_pct(v, decimals=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}%"


def fmt_ratio(v, decimals=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}x"


def cagr(start, end, years):
    if start is None or end is None or start <= 0 or end <= 0 or years <= 0:
        return None
    return ((end / start) ** (1 / years) - 1) * 100


# ─────────────────────────────────────────────────────────────────────────────
# EDGAR fetch logic  (ported straight from your dashboard, minimal changes)
# ─────────────────────────────────────────────────────────────────────────────

ANNUAL_FORMS = {"10-K", "10-K405", "10-KSB", "20-F", "20-F/A"}

# ── Full concept lists (mirrors the working tkinter dashboard) ────────────────
CF_CONCEPTS = {
    "NetCashProvidedByUsedInOperatingActivities": "OperatingCF",
    "NetCashProvidedByUsedInInvestingActivities": "InvestingCF",
    "NetCashProvidedByUsedInFinancingActivities": "FinancingCF",
    "PaymentsToAcquirePropertyPlantAndEquipment": "CapEx",
}
CF_CONCEPTS_IFRS = {
    "CashFlowsFromUsedInOperatingActivities":                             "OperatingCF",
    "CashFlowsFromOperations":                                            "OperatingCF",
    "CashFlowsFromUsedInInvestingActivities":                             "InvestingCF",
    "CashFlowsFromUsedInFinancingActivities":                             "FinancingCF",
    "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities": "CapEx",
    "AcquisitionOfPropertyPlantAndEquipment":                             "CapEx",
    "PurchaseOfPropertyPlantAndEquipment":                                "CapEx",
}
INC_CONCEPTS = {
    "Revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "RegulatedAndUnregulatedOperatingRevenue",
        "RevenuesNetOfInterestExpense",
    ],
    "CostOfSales": [
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsAndServiceExcludingDepletionDepreciationAndAmortization",
        "CostOfServices",
    ],
    "OperatingExpenses": [
        "OperatingExpenses",
        "CostsAndExpenses",
    ],
    "OperatingIncome": ["OperatingIncomeLoss"],
    "DA": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
        "DepreciationAmortizationAndAccretionNet",
    ],
    "InterestExpense": [
        "InterestExpense",
        "InterestAndDebtExpense",
        "InterestExpenseDebt",
    ],
    "TaxExpense": ["IncomeTaxExpense"],
    "PreTaxIncome": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ],
    "NetIncome": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
        "IncomeLossFromContinuingOperations",
        "NetIncomeLossAttributableToParent",
    ],
    "DilutedShares": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    ],
    "EPSDiluted": [
        "EarningsPerShareDiluted",
        "IncomeLossFromContinuingOperationsPerDilutedShare",
        "EarningsPerShareBasicAndDiluted",
    ],
}
INC_CONCEPTS_IFRS = {
    "Revenue": [
        "Revenue",
        "RevenueFromContractsWithCustomers",
        "RevenueFromSaleOfGoods",
        "RevenueFromRenderingOfServices",
    ],
    "CostOfSales": ["CostOfSales", "CostOfGoodsAndServicesRendered"],
    "OperatingIncome": [
        "ProfitLossFromOperatingActivities",
        "OperatingProfit",
        "ProfitLossBeforeFinanceCostsAndIncomeTax",
    ],
    "DA": [
        "DepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss",
        "DepreciationAndAmortisationExpense",
        "AdjustmentsForDepreciationAndAmortisationExpense",
    ],
    "InterestExpense": ["FinanceCosts", "InterestExpense"],
    "TaxExpense": ["IncomeTaxExpenseContinuingOperations", "IncomeTaxExpense"],
    "PreTaxIncome": ["ProfitLossBeforeTax"],
    "NetIncome": [
        "ProfitLossAttributableToOwnersOfParent",
        "ProfitLossAttributableToEquityHoldersOfParent",
        "ProfitLoss",
    ],
    "DilutedShares": [
        "WeightedAverageNumberOfSharesOutstandingDiluted",
        "WeightedAverageShares",
    ],
    "EPSDiluted": ["DilutedEarningsLossPerShare", "BasicEarningsLossPerShare"],
}
BS_CONCEPTS = {
    "Cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsAndShortTermInvestments",
        "CashAndDueFromBanks",
    ],
    "CurrentAssets":    ["AssetsCurrent"],
    "TotalAssets":      ["Assets"],
    "CurrentLiabilities": ["LiabilitiesCurrent"],
    "CurrentDebt": [
        "LongTermDebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "DebtCurrent",
        "ShortTermBorrowings",
        "CommercialPaper",
    ],
    "LongTermDebt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermNotesPayable",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "TotalLiabilities": ["Liabilities"],
    "TotalLiabilitiesAndEquity": ["LiabilitiesAndStockholdersEquity", "LiabilitiesAndPartnersCapital"],
    "StockholdersEquity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "MembersEquity",
    ],
}
BS_CONCEPTS_IFRS = {
    "Cash":             ["CashAndCashEquivalents"],
    "CurrentAssets":    ["CurrentAssets"],
    "TotalAssets":      ["Assets"],
    "CurrentLiabilities": ["CurrentLiabilities"],
    "CurrentDebt":      ["CurrentPortionOfNoncurrentBorrowings", "CurrentBorrowings"],
    "LongTermDebt":     ["NoncurrentPortionOfNoncurrentBorrowings", "NoncurrentBorrowings"],
    "TotalLiabilities": ["Liabilities"],
    "TotalLiabilitiesAndEquity": ["EquityAndLiabilities"],
    "StockholdersEquity": ["EquityAttributableToOwnersOfParent", "Equity"],
}


# ── Core EDGAR helpers (mirrors the working tkinter dashboard exactly) ────────

def _prep_dates(df):
    df = df.copy()
    # Guard against duplicate column names (can arrive from multi-source concat)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    for c in ("period_start", "period_end", "filing_date"):
        if c in df.columns:
            col = df[c]
            # If duplicate columns survived, col is a DataFrame — take first column
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            df[c] = pd.to_datetime(col, errors="coerce")
    return df


def _is_annual_form(series: pd.Series) -> pd.Series:
    s = series.fillna("").str.strip().str.upper()
    annual_upper = {f.upper() for f in ANNUAL_FORMS}
    return s.isin(annual_upper) | s.str.startswith("10-K") | s.str.startswith("20-F")


def _filter_annual_flow(df: pd.DataFrame) -> pd.DataFrame:
    df = _prep_dates(df)
    annual_mask = _is_annual_form(df["form_type"])
    ann = df[annual_mask].copy()

    # Primary: annual form rows with ~365-day duration
    if not ann.empty and "period_start" in ann.columns:
        days = (ann["period_end"] - ann["period_start"]).dt.days
        result = ann[(days >= 310) & (days <= 400)].copy()
        if not result.empty:
            return result

    # Fallback: duration-only filter on all rows (form_type may be unreliable)
    if "period_start" in df.columns and df["period_start"].notna().any():
        days = (df["period_end"] - df["period_start"]).dt.days
        result = df[(days >= 310) & (days <= 400)].copy()
        if not result.empty:
            return result

    # Last resort: all annual-form rows regardless of duration
    return ann if not ann.empty else df


def _filter_annual_instant(df: pd.DataFrame) -> pd.DataFrame:
    df = _prep_dates(df)
    annual_mask = _is_annual_form(df["form_type"])
    ann = df[annual_mask].copy()

    if not ann.empty and "period_start" in ann.columns:
        # True instant facts have no period_start
        result = ann[ann["period_start"].isna()].copy()
        if not result.empty:
            return result
        # Fallback: period_start == period_end (point-in-time)
        same = (ann["period_start"] - ann["period_end"]).dt.days.abs() <= 3
        result = ann[same].copy()
        if not result.empty:
            return result

    return ann if not ann.empty else df


def _dedup_pivot(df: pd.DataFrame, fact_cols: list) -> pd.DataFrame:
    df = df.copy()
    df["year_bucket"] = df["period_end"].dt.year
    df = (df.sort_values("filing_date")
           .drop_duplicates(subset=["fact", "year_bucket"], keep="last"))
    canonical_end = df.groupby("year_bucket")["period_end"].max()
    df["period_end"] = df["year_bucket"].map(canonical_end)
    pivot = df.pivot(index="period_end", columns="fact",
                     values="numeric_value").sort_index()
    for c in fact_cols:
        if c not in pivot.columns:
            pivot[c] = float("nan")
    return pivot[fact_cols]


def _build_from_concepts(df_annual: pd.DataFrame,
                          concepts: dict,
                          ifrs_concepts: dict | None = None) -> pd.DataFrame:
    """
    Build a pivot table from XBRL concept data.
    Tries both 'namespace:tag' and bare 'tag' forms so it works regardless
    of whether the namespace prefix was attached during extraction.
    """
    frames = []
    for metric, candidates in concepts.items():
        # Build both prefixed and bare lookup sets
        prefixed_gaap  = {f"us-gaap:{c}" for c in candidates}
        bare           = set(candidates)
        prefixed_ifrs: set = set()
        bare_ifrs: set = set()
        if ifrs_concepts and metric in ifrs_concepts:
            prefixed_ifrs = {f"ifrs-full:{c}" for c in ifrs_concepts[metric]}
            bare_ifrs     = set(ifrs_concepts[metric])
        all_names = prefixed_gaap | prefixed_ifrs | bare | bare_ifrs
        sub = df_annual[df_annual["concept"].isin(all_names)].copy()
        if not sub.empty:
            sub["fact"] = metric
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    pivot = _dedup_pivot(combined, list(concepts.keys()))
    pivot.index.name = "period_end"
    pivot.reset_index(inplace=True)
    return pivot


def _facts_to_df(facts) -> pd.DataFrame:
    """
    Convert an edgartools EntityFacts object to a normalised flat DataFrame.

    Column contract on return:
        form_type, period_end, period_start, numeric_value,
        accession_no, filing_date, concept, namespace

    where  concept = "namespace:tag"  (e.g. "us-gaap:Revenues")

    The function tries every known edgartools API shape so it works across
    versions 2.x → 3.x.
    """
    raw = None

    # ── Try every known method that returns a flat DataFrame ─────────────────
    for method_name in ("to_pandas", "to_dataframe"):
        if hasattr(facts, method_name):
            try:
                raw = getattr(facts, method_name)()
                if not isinstance(raw, pd.DataFrame) or raw.empty:
                    raw = None
                else:
                    break
            except Exception:
                raw = None

    # ── Try facts.facts  (dict or DataFrame, edgartools 3.x) ─────────────────
    if raw is None and hasattr(facts, "facts"):
        inner = facts.facts
        if isinstance(inner, pd.DataFrame) and not inner.empty:
            raw = inner.copy()
        elif isinstance(inner, dict):
            parts = []
            for ns, v in inner.items():
                if isinstance(v, pd.DataFrame) and not v.empty:
                    v = v.copy()
                    if "namespace" not in v.columns:
                        v["namespace"] = ns
                    parts.append(v)
                elif isinstance(v, dict):
                    # Nested dict: {namespace: {concept_name: list_of_dicts}}
                    for concept_tag, fact_list in v.items():
                        if isinstance(fact_list, list) and fact_list:
                            try:
                                df_c = pd.DataFrame(fact_list)
                                df_c["namespace"] = ns
                                df_c["fact"]      = concept_tag
                                parts.append(df_c)
                            except Exception:
                                pass
            if parts:
                raw = pd.concat(parts, ignore_index=True)

    # ── Iterate per-namespace (last resort) ───────────────────────────────────
    if raw is None:
        parts = []
        for ns in ("us-gaap", "ifrs-full", "dei"):
            try:
                ns_obj = facts[ns]
                if ns_obj is None:
                    continue
                for method_name in ("to_pandas", "to_dataframe"):
                    if hasattr(ns_obj, method_name):
                        try:
                            df_ns = getattr(ns_obj, method_name)()
                            if isinstance(df_ns, pd.DataFrame) and not df_ns.empty:
                                df_ns = df_ns.copy()
                                if "namespace" not in df_ns.columns:
                                    df_ns["namespace"] = ns
                                parts.append(df_ns)
                                break
                        except Exception:
                            pass
            except (KeyError, TypeError):
                pass
        if parts:
            raw = pd.concat(parts, ignore_index=True)

    if raw is None or raw.empty:
        raise ValueError(
            "Cannot extract facts from EntityFacts. "
            "Try: pip install --upgrade edgartools"
        )

    # ── Drop duplicate column names produced by pd.concat across sources ──────
    # When multiple parts are concatenated, the same column can appear twice
    # (e.g. two "namespace" columns). Downstream .str / arithmetic ops then
    # receive a 2-D DataFrame instead of a Series and crash.
    if raw.columns.duplicated().any():
        raw = raw.loc[:, ~raw.columns.duplicated(keep="first")]

    # ── Normalise column names ────────────────────────────────────────────────
    # Map every known alias → canonical name
    COL_MAP = {
        "form":          "form_type",
        "end":           "period_end",
        "start":         "period_start",
        "val":           "numeric_value",
        "value":         "numeric_value",
        "accn":          "accession_no",
        "accession":     "accession_no",
        "filed":         "filing_date",
        "filed_date":    "filing_date",
        "fact":          "concept",
        "tag":           "concept",
    }
    raw = raw.rename(columns={k: v for k, v in COL_MAP.items()
                               if k in raw.columns and k != v})

    # Ensure namespace column
    if "namespace" not in raw.columns:
        if "concept" in raw.columns:
            has_colon = raw["concept"].astype(str).str.contains(":", na=False)
            if has_colon.any():
                raw["namespace"] = (raw["concept"].astype(str)
                                    .str.split(":", n=1).str[0]
                                    .fillna("us-gaap"))
            else:
                raw["namespace"] = "us-gaap"
        else:
            raw["namespace"] = "us-gaap"

    # Attach "namespace:" prefix to bare concept names
    if "concept" in raw.columns:
        raw["concept"] = raw["concept"].astype(str).fillna("")
        no_prefix = ~raw["concept"].str.contains(":", na=False)
        if no_prefix.any():
            ns_series = raw.loc[no_prefix, "namespace"].fillna("us-gaap").astype(str)
            raw.loc[no_prefix, "concept"] = ns_series + ":" + raw.loc[no_prefix, "concept"]

    # Ensure all required columns exist
    for col, default in [
        ("period_start",  pd.NaT),
        ("accession_no",  ""),
        ("filing_date",   pd.NaT),
        ("form_type",     ""),
        ("numeric_value", np.nan),
    ]:
        if col not in raw.columns:
            raw[col] = default

    return raw



@st.cache_data(ttl=3600, show_spinner=False)
def fetch_edgar(ticker: str, email: str):
    """Return (income_df, bs_df, cf_df, quarterly_df, raw) or raise."""
    from edgar import set_identity, Company

    set_identity(email)
    company = Company(ticker)
    facts = company.get_facts()
    if facts is None:
        raise ValueError(f"No EDGAR facts for {ticker}")

    raw = _facts_to_df(facts)

    # ── Debug: stash raw info in cache for the UI to show ────────────────────
    _debug = {
        "rows":       len(raw),
        "cols":       list(raw.columns),
        "form_types": sorted(raw["form_type"].dropna().unique().tolist())[:30],
        "concept_sample": raw["concept"].dropna().head(10).tolist(),
    }

    flow_raw    = _filter_annual_flow(raw)
    instant_raw = _filter_annual_instant(raw)

    # ── Income statement ──────────────────────────────────────────────────────
    income_df = _build_from_concepts(flow_raw, INC_CONCEPTS, INC_CONCEPTS_IFRS)
    if income_df.empty:
        raise ValueError(
            f"No income data found.\n\n"
            f"DEBUG — raw rows: {_debug['rows']} | "
            f"columns: {_debug['cols']} | "
            f"form_types present: {_debug['form_types']} | "
            f"flow_raw rows after filter: {len(flow_raw)} | "
            f"sample concepts: {_debug['concept_sample']}"
        )

    if "DilutedShares" in income_df.columns:
        sh = income_df["DilutedShares"].dropna()
        if len(sh) >= 2 and sh.median() > 0:
            mask = income_df["DilutedShares"] < sh.median() * 0.01
            income_df.loc[mask, "DilutedShares"] *= 1000

    income_df["GrossProfit"]     = income_df["Revenue"] - income_df["CostOfSales"]
    income_df["GrossMargin"]     = income_df["GrossProfit"] / income_df["Revenue"] * 100
    income_df["OperatingMargin"] = income_df["OperatingIncome"] / income_df["Revenue"] * 100
    income_df["EBITDA"]          = income_df["OperatingIncome"] + income_df["DA"]
    income_df["EBITDAMargin"]    = income_df["EBITDA"] / income_df["Revenue"] * 100
    income_df = income_df.sort_values("period_end").reset_index(drop=True)
    income_df["RevenueGrowth"]       = income_df["Revenue"].pct_change() * 100
    income_df["OperatingIncomeGrowth"] = income_df["OperatingIncome"].pct_change() * 100
    income_df["NetIncomeGrowth"]     = income_df["NetIncome"].pct_change() * 100

    # ── Balance sheet ─────────────────────────────────────────────────────────
    bs_df = _build_from_concepts(instant_raw, BS_CONCEPTS, BS_CONCEPTS_IFRS)
    if not bs_df.empty:
        mask = bs_df["TotalLiabilities"].isna()
        bs_df.loc[mask, "TotalLiabilities"] = (
            bs_df.loc[mask, "TotalLiabilitiesAndEquity"] - bs_df.loc[mask, "StockholdersEquity"]
        )
        bs_df.drop(columns=["TotalLiabilitiesAndEquity"], inplace=True, errors="ignore")
        bs_df["TotalDebt"] = (
            bs_df.get("LongTermDebt", pd.Series(0, index=bs_df.index)).fillna(0) +
            bs_df.get("CurrentDebt", pd.Series(0, index=bs_df.index)).fillna(0)
        )
        bs_df["NetDebt"] = bs_df["TotalDebt"] - bs_df["Cash"].fillna(0)

    # ── Cash flow ─────────────────────────────────────────────────────────────
    cf_lk = {f"us-gaap:{k}": v for k, v in CF_CONCEPTS.items()}
    cf_lk.update({f"ifrs-full:{k}": v for k, v in CF_CONCEPTS_IFRS.items()})
    cf_df_raw = flow_raw[flow_raw["concept"].isin(cf_lk)].copy()
    if cf_df_raw.empty:
        raise ValueError("No cash flow data")
    cf_df_raw["fact"] = cf_df_raw["concept"].map(cf_lk)
    cf_df = _dedup_pivot(cf_df_raw, list(CF_CONCEPTS.values()))
    cf_df.index.name = "period_end"
    cf_df.reset_index(inplace=True)
    cf_df["FreeCashFlow"] = np.where(
        cf_df["CapEx"].notna(),
        cf_df["OperatingCF"] - cf_df["CapEx"],
        cf_df["OperatingCF"] + cf_df.get("InvestingCF", 0).fillna(0)
    )

    # ── Quarterly individual quarters ─────────────────────────────────────────
    quarterly_df = _build_quarterly(raw)

    return income_df, bs_df, cf_df, quarterly_df, raw


def _build_quarterly(raw):
    """
    Extract single-quarter (incremental) P&L and cash flows from YTD 10-Q filings.
    Returns a DataFrame with one row per quarter.
    """
    inc_lk = {f"us-gaap:{c}": m for m, cs in INC_CONCEPTS.items() for c in cs}
    inc_lk.update({f"ifrs-full:{c}": m for m, cs in INC_CONCEPTS_IFRS.items() for c in cs})
    cf_lk  = {f"us-gaap:{k}": v for k, v in CF_CONCEPTS.items()}
    cf_lk.update({f"ifrs-full:{k}": v for k, v in CF_CONCEPTS_IFRS.items()})
    all_lk = {**inc_lk, **cf_lk}

    ann = _prep_dates(raw[raw["form_type"].isin(ANNUAL_FORMS)].copy())
    ann = ann[ann["period_start"].notna()].copy()
    ann["dur"] = (ann["period_end"] - ann["period_start"]).dt.days
    ann = ann[(ann["dur"] >= 340) & (ann["dur"] <= 390)].copy()
    ann = ann[ann["concept"].isin(all_lk)].copy()
    ann["fact"]   = ann["concept"].map(all_lk)
    ann["q_slot"] = 4

    q = _prep_dates(raw[raw["form_type"] == "10-Q"].copy())
    q = q[q["period_start"].notna()].copy()
    q["dur"] = (q["period_end"] - q["period_start"]).dt.days
    q = q[(q["dur"] >= 75) & (q["dur"] <= 285)].copy()
    q = q[q["concept"].isin(all_lk)].copy()
    q["fact"]   = q["concept"].map(all_lk)
    q["q_slot"] = (q["dur"] / 91).round().clip(1, 3).astype(int)

    if ann.empty and q.empty:
        return pd.DataFrame()

    # Use annual period-end dates to assign fiscal year
    ann_ends = sorted(_prep_dates(ann)["period_end"].dropna().unique()) if not ann.empty else []

    def _assign_fy(pe):
        for ae in ann_ends:
            if pe <= ae:
                return ae.year
        return pe.year

    combined = pd.concat([ann, q], ignore_index=True)
    if ann_ends:
        combined["fy"] = combined["period_end"].apply(_assign_fy)
    else:
        combined["fy"] = combined["period_end"].dt.year

    combined = (combined.sort_values("filing_date")
                .drop_duplicates(subset=["fact","fy","q_slot"], keep="last"))

    pivot = combined.pivot_table(
        index=["fy","q_slot","period_end"], columns="fact",
        values="numeric_value", aggfunc="last"
    ).reset_index()

    flow_cols = [c for c in list(set(list(INC_CONCEPTS.keys()) + list(CF_CONCEPTS.values())))
                 if c in pivot.columns]

    rows = []
    for fy, grp in pivot.groupby("fy"):
        grp = grp.sort_values("q_slot").reset_index(drop=True)
        for i, row in grp.iterrows():
            q_slot = int(row["q_slot"])
            prev   = grp[grp["q_slot"] < q_slot]
            new_row = {"fy": fy, "quarter": q_slot, "period_end": row["period_end"]}
            for col in flow_cols:
                ytd = _safe(row.get(col))
                if prev.empty:
                    new_row[col] = ytd
                else:
                    prev_ytd = _safe(prev.iloc[-1].get(col))
                    new_row[col] = (ytd - prev_ytd if ytd is not None and prev_ytd is not None else ytd)
            rows.append(new_row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values(["fy","quarter"]).reset_index(drop=True)

    if "Revenue" in result.columns and "CostOfSales" in result.columns:
        result["GrossProfit"]     = result["Revenue"] - result["CostOfSales"]
        result["GrossMargin"]     = result["GrossProfit"] / result["Revenue"] * 100
    if "OperatingIncome" in result.columns and "Revenue" in result.columns:
        result["OperatingMargin"] = result["OperatingIncome"] / result["Revenue"] * 100
    if "OperatingIncome" in result.columns and "DA" in result.columns:
        result["EBITDA"]      = result["OperatingIncome"] + result["DA"]
        result["EBITDAMargin"] = result["EBITDA"] / result["Revenue"] * 100
    if "OperatingCF" in result.columns and "CapEx" in result.columns:
        result["FreeCashFlow"] = result["OperatingCF"] - result["CapEx"].fillna(0)

    result["period_label"] = "Q" + result["quarter"].astype(str) + " " + result["fy"].astype(str)
    result["period_end"]   = result["period_end"].astype(str)
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(ticker: str):
    try:
        stk  = yf.Ticker(ticker)
        hist = stk.history(period="max", auto_adjust=True)
        if not hist.empty:
            hist.index = hist.index.tz_localize(None)
        info = stk.info
        return hist, info
    except Exception:
        return pd.DataFrame(), {}


# ─────────────────────────────────────────────────────────────────────────────
# TTM computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ttm(income_df, bs_df, cf_df):
    """Return dict of TTM values for key metrics."""
    ttm = {}
    if income_df.empty:
        return ttm

    # Pick latest 4 annual rows (one per year) to find the most recent annual filing
    latest_annual = income_df.sort_values("period_end").iloc[-1]
    ttm["Revenue"]          = _safe(latest_annual.get("Revenue"))
    ttm["GrossProfit"]      = _safe(latest_annual.get("GrossProfit"))
    ttm["OperatingIncome"]  = _safe(latest_annual.get("OperatingIncome"))
    ttm["NetIncome"]        = _safe(latest_annual.get("NetIncome"))
    ttm["EBITDA"]           = _safe(latest_annual.get("EBITDA"))
    ttm["GrossMargin"]      = _safe(latest_annual.get("GrossMargin"))
    ttm["OperatingMargin"]  = _safe(latest_annual.get("OperatingMargin"))
    ttm["EBITDAMargin"]     = _safe(latest_annual.get("EBITDAMargin"))
    ttm["DilutedShares"]    = _safe(latest_annual.get("DilutedShares"))
    ttm["EPSDiluted"]       = _safe(latest_annual.get("EPSDiluted"))
    ttm["DA"]               = _safe(latest_annual.get("DA"))

    if not cf_df.empty:
        latest_cf = cf_df.sort_values("period_end").iloc[-1]
        ttm["OperatingCF"]  = _safe(latest_cf.get("OperatingCF"))
        ttm["CapEx"]        = _safe(latest_cf.get("CapEx"))
        ttm["FreeCashFlow"] = _safe(latest_cf.get("FreeCashFlow"))

    if not bs_df.empty:
        latest_bs = bs_df.sort_values("period_end").iloc[-1]
        ttm["Cash"]                = _safe(latest_bs.get("Cash"))
        ttm["TotalDebt"]           = _safe(latest_bs.get("TotalDebt"))
        ttm["NetDebt"]             = _safe(latest_bs.get("NetDebt"))
        ttm["StockholdersEquity"]  = _safe(latest_bs.get("StockholdersEquity"))
        ttm["TotalAssets"]         = _safe(latest_bs.get("TotalAssets"))
        ttm["TotalLiabilities"]    = _safe(latest_bs.get("TotalLiabilities"))

    return ttm


# ─────────────────────────────────────────────────────────────────────────────
# Valuation ratios
# ─────────────────────────────────────────────────────────────────────────────

def compute_valuation(ttm, price, mkt_cap):
    v = {}
    sh = ttm.get("DilutedShares")
    if price and sh and sh > 0:
        ni  = ttm.get("NetIncome")
        rev = ttm.get("Revenue")
        fcf = ttm.get("FreeCashFlow")
        ebitda = ttm.get("EBITDA")
        ocf = ttm.get("OperatingCF")
        eq  = ttm.get("StockholdersEquity")
        gp  = ttm.get("GrossProfit")

        eps = ni / sh if ni else None
        if eps and eps > 0:
            v["P/E"]   = price / eps
        rev_ps = rev / sh if rev else None
        if rev_ps and rev_ps > 0:
            v["P/S"]   = price / rev_ps
        fcf_ps = fcf / sh if fcf else None
        if fcf_ps and fcf_ps > 0:
            v["P/FCF"] = price / fcf_ps
        ocf_ps = ocf / sh if ocf else None
        if ocf_ps and ocf_ps > 0:
            v["P/CFO"] = price / ocf_ps
        gp_ps = gp / sh if gp else None
        if gp_ps and gp_ps > 0:
            v["P/GP"]  = price / gp_ps
        eq_ps = eq / sh if eq else None
        if eq_ps and eq_ps > 0:
            v["P/Book"] = price / eq_ps

    if mkt_cap:
        td  = ttm.get("TotalDebt") or 0
        cas = ttm.get("Cash") or 0
        ev  = mkt_cap + td - cas
        rev = ttm.get("Revenue")
        ebitda = ttm.get("EBITDA")
        if rev and rev > 0:
            v["EV/Sales"]  = ev / rev
        if ebitda and ebitda > 0:
            v["EV/EBITDA"] = ev / ebitda
        v["_ev"] = ev

    return v


# ─────────────────────────────────────────────────────────────────────────────
# CAGR table
# ─────────────────────────────────────────────────────────────────────────────

def build_cagr_table(income_df, cf_df):
    """Return DataFrame with 1-14-yr CAGRs for key metrics."""
    if income_df.empty:
        return pd.DataFrame()

    sh = income_df["DilutedShares"].replace(0, np.nan)
    df = income_df.sort_values("period_end").copy()

    series_map = {
        "Revenue/Share":    (df["Revenue"] / sh).values,
        "EPS (NI/Share)":   (df["NetIncome"] / sh).values,
        "GP/Share":         (df.get("GrossProfit", pd.Series(0, index=df.index)) / sh).values,
        "EBITDA/Share":     (df.get("EBITDA", pd.Series(0, index=df.index)) / sh).values,
    }
    if not cf_df.empty and "FreeCashFlow" in cf_df.columns:
        cf_a = cf_df.set_index("period_end")["FreeCashFlow"].reindex(df["period_end"].values)
        series_map["FCF/Share"] = (cf_a.values / sh.values)

    years_range = list(range(1, 15))
    rows = []
    for label, vals in series_map.items():
        row = {"Metric": label}
        for yr in years_range:
            n = yr  # annual rows, 1 row = 1 year
            if len(vals) >= n + 1:
                start = vals[-(n + 1)]
                end   = vals[-1]
                try:
                    start = float(start); end = float(end)
                    c = cagr(start, end, yr)
                    row[f"{yr}Y"] = f"{c:.1f}%" if c is not None else "—"
                except Exception:
                    row[f"{yr}Y"] = "—"
            else:
                row[f"{yr}Y"] = "—"
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plotly helpers
# ─────────────────────────────────────────────────────────────────────────────

PLOT_COLORS = ["#00A0A0","#FFA500","#6BCB77","#FF6B6B","#7B9CFF","#C77DFF","#FFD166"]

def bar_chart(df, x_col, y_cols, title, yaxis_title="$B"):
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce") / 1e9  # to billions
        fig.add_trace(go.Bar(
            x=df[x_col].astype(str), y=vals,
            name=col, marker_color=PLOT_COLORS[i % len(PLOT_COLORS)],
        ))
    fig.update_layout(
        title=title, barmode="group",
        plot_bgcolor="#1e2433", paper_bgcolor="#1e2433",
        font=dict(color="#c9d1d9", size=12),
        xaxis=dict(gridcolor="#2e3550", tickangle=-30),
        yaxis=dict(gridcolor="#2e3550", title=yaxis_title),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=360, margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig


def line_chart(df, x_col, y_cols, title, yaxis_title="%"):
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        fig.add_trace(go.Scatter(
            x=df[x_col].astype(str), y=vals,
            mode="lines+markers", name=col,
            line=dict(color=PLOT_COLORS[i % len(PLOT_COLORS)], width=2),
            marker=dict(size=6),
        ))
    fig.update_layout(
        title=title,
        plot_bgcolor="#1e2433", paper_bgcolor="#1e2433",
        font=dict(color="#c9d1d9", size=12),
        xaxis=dict(gridcolor="#2e3550", tickangle=-30),
        yaxis=dict(gridcolor="#2e3550", title=yaxis_title),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=360, margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig


def price_chart(hist: pd.DataFrame, ticker: str):
    if hist.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        mode="lines", name="Close",
        line=dict(color="#00A0A0", width=1.8),
        fill="tozeroy", fillcolor="rgba(0,160,160,0.08)",
    ))
    fig.update_layout(
        title=f"{ticker.upper()} — Price History",
        plot_bgcolor="#1e2433", paper_bgcolor="#1e2433",
        font=dict(color="#c9d1d9", size=12),
        xaxis=dict(gridcolor="#2e3550"),
        yaxis=dict(gridcolor="#2e3550", title="Price ($)"),
        height=300, margin=dict(t=50, b=30, l=50, r=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Metric card helper
# ─────────────────────────────────────────────────────────────────────────────

def mc(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("📊 EDGAR Dashboard")
st.sidebar.markdown("---")

ticker_input = st.sidebar.text_input("Ticker Symbol", value="AAPL", max_chars=10).strip().upper()
email_input  = st.sidebar.text_input(
    "SEC EDGAR Email (required)",
    value="your@email.com",
    help="SEC requires an identity string per https://efts.sec.gov/LATEST/search-index?q=%22fair+use%22",
)

fetch_btn = st.sidebar.button("🔍 Fetch Data", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Display window**")
n_annual   = st.sidebar.slider("Annual years to show",   min_value=3,  max_value=99, value=99, step=1)
n_quarters = st.sidebar.slider("Quarters to show",       min_value=4,  max_value=99, value=99, step=1)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**About**
- All financial data pulled live from **SEC EDGAR** via `edgartools`
- Price / market cap from **yfinance**
- No Macrotrends dependency

**Sections**
- 📌 Overview & Valuation
- 📈 Income Statement
- 🏦 Balance Sheet
- 💵 Cash Flow
- 📅 Quarterly Breakdown
- 📊 CAGR Tables
""")

# ─────────────────────────────────────────────────────────────────────────────
# Main panel
# ─────────────────────────────────────────────────────────────────────────────

st.title(f"SEC EDGAR Financial Dashboard")

if not fetch_btn and "last_ticker" not in st.session_state:
    st.info("Enter a ticker on the left and click **Fetch Data** to begin.")
    st.stop()

# Use session state to avoid re-fetch on widget interaction
if fetch_btn:
    st.session_state["last_ticker"] = ticker_input
    st.session_state["last_email"]  = email_input
    # clear caches for new ticker
    fetch_edgar.clear()
    fetch_price_data.clear()

ticker = st.session_state.get("last_ticker", ticker_input)
email  = st.session_state.get("last_email",  email_input)

if not email or "@" not in email:
    st.warning("Please enter a valid email address for SEC EDGAR identity.")
    st.stop()

# ── Fetch ─────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching EDGAR data for **{ticker}** …"):
    try:
        income_df, bs_df, cf_df, quarterly_df, raw = fetch_edgar(ticker, email)
    except Exception as err:
        st.error(f"EDGAR fetch failed: {err}")
        with st.expander("🔍 Raw debug info (paste this if asking for help)"):
            st.code(str(err), language="text")
        st.stop()

# Debug expander — always shown so you can verify raw data shape
with st.expander("🛠 Raw EDGAR data shape (click to expand)", expanded=False):
    if not raw.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total rows", f"{len(raw):,}")
        c2.metric("Columns", len(raw.columns))
        c3.metric("Annual flow rows", f"{len(_filter_annual_flow(raw)):,}")
        st.write("**Columns:**", list(raw.columns))
        form_counts = raw["form_type"].value_counts().head(20)
        st.write("**Form types (top 20):**", form_counts.to_dict())
        st.write("**Sample concepts (first 10):**", raw["concept"].dropna().head(10).tolist())

with st.spinner("Fetching price data …"):
    price_hist, yf_info = fetch_price_data(ticker)

# ── Derived ───────────────────────────────────────────────────────────────────
ttm      = compute_ttm(income_df, bs_df, cf_df)
price    = _safe(yf_info.get("currentPrice") or yf_info.get("regularMarketPrice"))
mkt_cap  = _safe(yf_info.get("marketCap"))
val      = compute_valuation(ttm, price, mkt_cap)
ev       = val.pop("_ev", None)
cagr_df  = build_cagr_table(income_df, cf_df)

company_name = yf_info.get("longName") or yf_info.get("shortName") or ticker
sector       = yf_info.get("sector", "—")
industry     = yf_info.get("industry", "—")

# ─────────────────────────────────────────────────────────────────────────────
# Header strip
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"## {company_name} &nbsp; `{ticker}`")
st.caption(f"{sector} › {industry}")

c1, c2, c3, c4, c5 = st.columns(5)
with c1: mc("Price",      f"${price:.2f}" if price else "—")
with c2: mc("Market Cap", fmt_val(mkt_cap))
with c3: mc("Revenue TTM", fmt_val(ttm.get("Revenue")))
with c4: mc("Net Income TTM", fmt_val(ttm.get("NetIncome")))
with c5: mc("FCF TTM", fmt_val(ttm.get("FreeCashFlow")))

c1, c2, c3, c4, c5 = st.columns(5)
with c1: mc("Gross Margin", fmt_pct(ttm.get("GrossMargin")))
with c2: mc("Operating Margin", fmt_pct(ttm.get("OperatingMargin")))
with c3: mc("EBITDA TTM", fmt_val(ttm.get("EBITDA")))
with c4: mc("EPS Diluted", f"${ttm.get('EPSDiluted'):.2f}" if ttm.get("EPSDiluted") else "—")
with c5: mc("Net Debt", fmt_val(ttm.get("NetDebt")))

# ─────────────────────────────────────────────────────────────────────────────
# Price chart
# ─────────────────────────────────────────────────────────────────────────────
if not price_hist.empty:
    fig_price = price_chart(price_hist, ticker)
    if fig_price:
        st.plotly_chart(fig_price, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📌 Valuation",
    "📈 Income",
    "🏦 Balance Sheet",
    "💵 Cash Flow",
    "📅 Quarterly",
    "📊 CAGRs",
    "🗂 Raw XBRL",
])

# ── Tab 0: Valuation ─────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-header">Current Valuation Multiples</div>', unsafe_allow_html=True)

    if val:
        cols = st.columns(min(len(val), 4))
        for i, (k, v) in enumerate(val.items()):
            with cols[i % 4]:
                mc(k, f"{v:.2f}x")

    st.markdown('<div class="section-header">Enterprise Value Components</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: mc("Enterprise Value", fmt_val(ev))
    with c2: mc("Total Debt", fmt_val(ttm.get("TotalDebt")))
    with c3: mc("Cash & Equivalents", fmt_val(ttm.get("Cash")))

    # EV / multiple table
    ev_rows = []
    if ev:
        for label, key in [("EV/Revenue", "Revenue"), ("EV/EBITDA", "EBITDA"),
                           ("EV/FCF", "FreeCashFlow"), ("EV/OCF", "OperatingCF")]:
            denom = ttm.get(key)
            if denom and denom > 0:
                ev_rows.append({"Multiple": label, "Value": f"{ev/denom:.2f}x"})
    if ev_rows:
        st.dataframe(pd.DataFrame(ev_rows), hide_index=True, use_container_width=False)

    st.markdown('<div class="section-header">EPS & Growth Context</div>', unsafe_allow_html=True)
    eps = ttm.get("EPSDiluted")
    ni  = ttm.get("NetIncome")
    sh  = ttm.get("DilutedShares")
    if not eps and ni and sh and sh > 0:
        eps = ni / sh

    pe = val.get("P/E")
    fcf_ps = (ttm.get("FreeCashFlow") / sh) if (ttm.get("FreeCashFlow") and sh and sh > 0) else None

    c1, c2, c3, c4 = st.columns(4)
    with c1: mc("EPS (TTM)", f"${eps:.2f}" if eps else "—")
    with c2: mc("FCF/Share", f"${fcf_ps:.2f}" if fcf_ps else "—")
    with c3: mc("Shares Diluted", f"{sh/1e9:.3f}B" if sh and sh > 1e8 else f"{sh/1e6:.1f}M" if sh else "—")

    # Revenue/EPS growth trend (last 5 years)
    if len(income_df) >= 2:
        st.markdown('<div class="section-header">Revenue & Net Income Growth (Annual)</div>', unsafe_allow_html=True)
        df_plot = income_df.tail(n_annual).copy()
        df_plot["period_end_str"] = df_plot["period_end"].astype(str).str[:4]
        fig_g = line_chart(
            df_plot, "period_end_str",
            ["RevenueGrowth", "NetIncomeGrowth"],
            "YoY Growth %", yaxis_title="YoY %"
        )
        st.plotly_chart(fig_g, use_container_width=True)


# ── Tab 1: Income Statement ───────────────────────────────────────────────────
with tabs[1]:
    if income_df.empty:
        st.warning("No income statement data available.")
    else:
        df_i = income_df.copy()
        df_i["Year"] = df_i["period_end"].astype(str).str[:4]

        st.markdown('<div class="section-header">Revenue, Gross Profit, Operating Income, Net Income</div>', unsafe_allow_html=True)
        fig_inc = bar_chart(df_i.tail(n_annual), "Year",
                            ["Revenue","GrossProfit","OperatingIncome","NetIncome"],
                            "Income Statement ($B)")
        st.plotly_chart(fig_inc, use_container_width=True)

        st.markdown('<div class="section-header">Margins</div>', unsafe_allow_html=True)
        fig_mg = line_chart(df_i.tail(n_annual), "Year",
                            ["GrossMargin","OperatingMargin","EBITDAMargin"],
                            "Margins %")
        st.plotly_chart(fig_mg, use_container_width=True)

        # Table
        display_cols = ["Year","Revenue","GrossProfit","OperatingIncome","EBITDA","NetIncome",
                        "GrossMargin","OperatingMargin","EBITDAMargin","DilutedShares","EPSDiluted",
                        "RevenueGrowth","NetIncomeGrowth"]
        display_cols = [c for c in display_cols if c in df_i.columns]
        st.markdown('<div class="section-header">Annual Data Table</div>', unsafe_allow_html=True)
        fmt_cols = {
            "Revenue": "${:,.0f}", "GrossProfit": "${:,.0f}",
            "OperatingIncome": "${:,.0f}", "EBITDA": "${:,.0f}", "NetIncome": "${:,.0f}",
            "GrossMargin": "{:.1f}%", "OperatingMargin": "{:.1f}%", "EBITDAMargin": "{:.1f}%",
            "RevenueGrowth": "{:.1f}%", "NetIncomeGrowth": "{:.1f}%",
            "DilutedShares": "{:,.0f}", "EPSDiluted": "${:.2f}",
        }
        tbl = df_i[display_cols].sort_values("Year", ascending=False).head(n_annual)
        st.dataframe(tbl, hide_index=True, use_container_width=True)


# ── Tab 2: Balance Sheet ──────────────────────────────────────────────────────
with tabs[2]:
    if bs_df.empty:
        st.warning("No balance sheet data available.")
    else:
        df_b = bs_df.copy()
        df_b["Year"] = df_b["period_end"].astype(str).str[:4]

        st.markdown('<div class="section-header">Assets vs Liabilities vs Equity</div>', unsafe_allow_html=True)
        fig_bs = bar_chart(df_b.tail(n_annual), "Year",
                           ["TotalAssets","TotalLiabilities","StockholdersEquity"],
                           "Balance Sheet ($B)")
        st.plotly_chart(fig_bs, use_container_width=True)

        st.markdown('<div class="section-header">Debt & Cash</div>', unsafe_allow_html=True)
        fig_dc = bar_chart(df_b.tail(n_annual), "Year",
                           ["TotalDebt","Cash","NetDebt"],
                           "Debt vs Cash ($B)")
        st.plotly_chart(fig_dc, use_container_width=True)

        disp = ["Year","Cash","CurrentAssets","TotalAssets","CurrentLiabilities",
                "LongTermDebt","TotalDebt","NetDebt","StockholdersEquity","TotalLiabilities"]
        disp = [c for c in disp if c in df_b.columns]
        st.dataframe(df_b[disp].sort_values("Year", ascending=False).head(n_annual), hide_index=True, use_container_width=True)


# ── Tab 3: Cash Flow ──────────────────────────────────────────────────────────
with tabs[3]:
    if cf_df.empty:
        st.warning("No cash flow data available.")
    else:
        df_c = cf_df.copy()
        df_c["Year"] = df_c["period_end"].astype(str).str[:4]

        st.markdown('<div class="section-header">Operating CF, CapEx, Free Cash Flow</div>', unsafe_allow_html=True)
        fig_cf = bar_chart(df_c.tail(n_annual), "Year",
                           ["OperatingCF","CapEx","FreeCashFlow"],
                           "Cash Flow ($B)")
        st.plotly_chart(fig_cf, use_container_width=True)

        # FCF vs Net Income comparison
        if not income_df.empty and "NetIncome" in income_df.columns:
            df_cmp = pd.merge(
                df_c[["Year","FreeCashFlow","OperatingCF"]],
                income_df.assign(Year=income_df["period_end"].astype(str).str[:4])[["Year","NetIncome"]],
                on="Year", how="inner",
            )
            if not df_cmp.empty:
                st.markdown('<div class="section-header">FCF vs Net Income (Sloan Signal)</div>', unsafe_allow_html=True)
                fig_sloan = bar_chart(df_cmp.tail(n_annual), "Year",
                                      ["FreeCashFlow","NetIncome"],
                                      "FCF vs Net Income — divergence = accrual signal ($B)")
                st.plotly_chart(fig_sloan, use_container_width=True)

        disp = ["Year","OperatingCF","InvestingCF","FinancingCF","CapEx","FreeCashFlow"]
        disp = [c for c in disp if c in df_c.columns]
        st.dataframe(df_c[disp].sort_values("Year", ascending=False).head(n_annual), hide_index=True, use_container_width=True)


# ── Tab 4: Quarterly Breakdown ────────────────────────────────────────────────
with tabs[4]:
    if quarterly_df.empty:
        st.warning("No quarterly data available.")
    else:
        q = quarterly_df.copy()

        # ── Revenue + Gross Profit ───────────────────────────────────────────
        st.markdown('<div class="section-header">Revenue & Gross Profit</div>', unsafe_allow_html=True)
        rev_cols = [c for c in ["Revenue","GrossProfit"] if c in q.columns]
        if rev_cols:
            fig_qrev = bar_chart(q.tail(n_quarters), "period_label", rev_cols,
                                 "Quarterly Revenue & Gross Profit ($B)")
            st.plotly_chart(fig_qrev, use_container_width=True)

        # ── Operating & Net Income ───────────────────────────────────────────
        inc_cols = [c for c in ["OperatingIncome","NetIncome"] if c in q.columns]
        if inc_cols:
            fig_qni = bar_chart(q.tail(n_quarters), "period_label", inc_cols,
                                "Quarterly Operating & Net Income ($B)")
            st.plotly_chart(fig_qni, use_container_width=True)

        # ── YoY Revenue growth (same quarter prior year) ─────────────────────
        if "Revenue" in q.columns:
            st.markdown('<div class="section-header">Revenue YoY Growth (same quarter)</div>', unsafe_allow_html=True)
            q_sorted = q.sort_values(["quarter","fy"]).copy()
            q_sorted["RevYoY"] = q_sorted.groupby("quarter")["Revenue"].pct_change() * 100
            q_sorted = q_sorted.sort_values(["fy","quarter"])
            fig_yoy = go.Figure()
            yoy_tail = q_sorted.tail(n_quarters)
            colors_yoy = ["#6BCB77" if v >= 0 else "#FF6B6B"
                          for v in yoy_tail["RevYoY"].fillna(0)]
            fig_yoy.add_trace(go.Bar(
                x=yoy_tail["period_label"], y=yoy_tail["RevYoY"],
                marker_color=colors_yoy, name="Rev YoY %",
            ))
            fig_yoy.update_layout(
                title="Revenue YoY Growth %",
                plot_bgcolor="#1e2433", paper_bgcolor="#1e2433",
                font=dict(color="#c9d1d9", size=12),
                xaxis=dict(gridcolor="#2e3550", tickangle=-30),
                yaxis=dict(gridcolor="#2e3550", title="YoY %"),
                height=350, margin=dict(t=50, b=60),
            )
            st.plotly_chart(fig_yoy, use_container_width=True)

        # ── Margins ─────────────────────────────────────────────────────────
        mg_cols = [c for c in ["GrossMargin","OperatingMargin","EBITDAMargin"] if c in q.columns]
        if mg_cols:
            st.markdown('<div class="section-header">Quarterly Margins</div>', unsafe_allow_html=True)
            fig_qmg = line_chart(q.tail(n_quarters), "period_label", mg_cols, "Quarterly Margins %")
            st.plotly_chart(fig_qmg, use_container_width=True)

        # ── Cash Flow ────────────────────────────────────────────────────────
        cf_cols = [c for c in ["OperatingCF","FreeCashFlow","CapEx"] if c in q.columns]
        if cf_cols:
            st.markdown('<div class="section-header">Quarterly Cash Flow</div>', unsafe_allow_html=True)
            fig_qcf = bar_chart(q.tail(n_quarters), "period_label", cf_cols,
                                "Quarterly Cash Flow ($B)")
            st.plotly_chart(fig_qcf, use_container_width=True)

        # ── Data table ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Quarterly Data Table</div>', unsafe_allow_html=True)
        display_q = [c for c in [
            "period_label","Revenue","GrossProfit","GrossMargin",
            "OperatingIncome","OperatingMargin","EBITDA","EBITDAMargin",
            "NetIncome","OperatingCF","CapEx","FreeCashFlow",
        ] if c in q.columns]
        st.dataframe(
            q[display_q].sort_values("period_label", ascending=False).head(n_quarters),
            hide_index=True, use_container_width=True,
        )


# ── Tab 5: CAGR Tables ───────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("CAGR calculated from the most recent annual value back N years, on a **per-share** basis.")

    if not cagr_df.empty:
        year_cols = [c for c in cagr_df.columns if c != "Metric"]
        st.dataframe(cagr_df.set_index("Metric")[year_cols], use_container_width=True)

        # Heatmap
        try:
            hm_data = cagr_df.set_index("Metric")[year_cols].copy()
            for c in hm_data.columns:
                hm_data[c] = hm_data[c].str.rstrip("%").replace("—", np.nan).astype(float)

            fig_hm = px.imshow(
                hm_data,
                color_continuous_scale=["#ff6b6b","#1e2433","#6bcb77"],
                zmin=-20, zmax=40,
                text_auto=".1f",
                title="CAGR Heatmap (%, per share)",
                aspect="auto",
            )
            fig_hm.update_layout(
                plot_bgcolor="#1e2433", paper_bgcolor="#1e2433",
                font=dict(color="#c9d1d9"), height=400,
            )
            st.plotly_chart(fig_hm, use_container_width=True)
        except Exception as e:
            st.caption(f"Heatmap skipped: {e}")


# ── Tab 6: Raw XBRL ───────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("Raw annual XBRL facts from SEC EDGAR, split into flow (income/CF) and instant (balance sheet).")

    raw_sub = tabs[6]

    col_flow, col_inst = st.columns(2)

    def _pivot_all(df_in):
        if df_in.empty:
            return pd.DataFrame()
        df_in = df_in.copy()
        df_in["label"] = df_in["concept"].str.split(":").str[-1]
        df_in["year_bucket"] = df_in["period_end"].dt.year
        df_in = df_in.sort_values("filing_date").drop_duplicates(subset=["label","year_bucket"], keep="last")
        canonical_end = df_in.groupby("year_bucket")["period_end"].max()
        df_in["period_end"] = df_in["year_bucket"].map(canonical_end)
        pivot = df_in.pivot_table(index="period_end", columns="label",
                                  values="numeric_value", aggfunc="last")
        pivot = pivot.dropna(axis=1, how="all").sort_index()
        pivot.index = pivot.index.astype(str).str[:4]
        return pivot.T.reset_index().rename(columns={"label": "Concept"})

    flow_raw_df    = _filter_annual_flow(raw)
    instant_raw_df = _filter_annual_instant(raw)

    with col_flow:
        st.markdown("**Flow (Income / Cash Flow)**")
        pf = _pivot_all(flow_raw_df)
        if not pf.empty:
            search_flow = st.text_input("Filter flow concepts", key="flow_search")
            if search_flow:
                pf = pf[pf["Concept"].str.contains(search_flow, case=False, na=False)]
            st.dataframe(pf, hide_index=True, use_container_width=True, height=600)

    with col_inst:
        st.markdown("**Instant (Balance Sheet)**")
        pi = _pivot_all(instant_raw_df)
        if not pi.empty:
            search_inst = st.text_input("Filter BS concepts", key="inst_search")
            if search_inst:
                pi = pi[pi["Concept"].str.contains(search_inst, case=False, na=False)]
            st.dataframe(pi, hide_index=True, use_container_width=True, height=600)
