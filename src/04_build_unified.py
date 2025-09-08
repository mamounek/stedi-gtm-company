import pandas as pd
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / "data" / "raw" / "companies.csv"
DEAL = ROOT / "data" / "raw" / "deals.csv"
BILL = ROOT / "data" / "raw" / "billing.csv"
OUT  = ROOT / "data" / "final" / "unified.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Load data
companies = pd.read_csv(COMP, parse_dates=["created_date"])
deals = pd.read_csv(
    DEAL,
    parse_dates=[
        "created_date",
        "stage_date_discovery","stage_date_evaluation",
        "stage_date_proposal","stage_date_negotiation",
        "stage_date_closed_won","stage_date_closed_lost",
    ],
)
billing = pd.read_csv(
    BILL,
    parse_dates=["billing_start_date","billing_end_date"]
)

# Join companies with deals by account_id
df = companies.merge(deals, on="account_id", how="left")

# Outcome flags
df["is_opportunity"] = df["deal_id"].notna()
df["is_customer"]    = df["stage_date_closed_won"].notna()
df["outcome"] = np.where(
    df["is_customer"], "won",
    np.where(df["stage_date_closed_lost"].notna(), "lost",
             np.where(df["is_opportunity"], "open", "no_opp"))
)

# Sales cycle (days) from deal.created_date to the latest stage date (won/lost/open)
stage_cols = [
    "stage_date_discovery","stage_date_evaluation","stage_date_proposal",
    "stage_date_negotiation","stage_date_closed_won","stage_date_closed_lost"
]
latest_stage = df[stage_cols].max(axis=1)
df["sales_cycle_days"] = (
    (latest_stage - df["created_date_y"])
    .dt.total_seconds().div(86400).round().astype("Int64")
)

# Join with billing by account_id and deal_id
bill_keep = [
    "billing_id","deal_id","account_id","billing_start_date","billing_end_date",
    "term_months","billing_frequency","arr","mrr","currency"
]
df = df.merge(billing[bill_keep], on=["account_id","deal_id"], how="left", suffixes=("",""))

# Select minimal columns for analysis
cols = [
    # Company/enrichment
    "account_id","account_name","account_domain","industry","country","employee_band",
    "created_date_x","source","campaign",
    "icp","icp_confidence","is_prev_stedi_customer",
    "health_tech","apis_cloud","integration_patterns","tech_stack_confidence",
    # Deal snapshot
    "deal_id","owner","created_date_y",
    "stage_date_discovery","stage_date_evaluation","stage_date_proposal","stage_date_negotiation",
    "stage_date_closed_won","stage_date_closed_lost",
    "sales_cycle_days","is_opportunity","is_customer","outcome",
    # Billing summary (annual)
    "billing_id","billing_start_date","billing_end_date","term_months","billing_frequency",
    "arr","mrr","currency",
]
present = [c for c in cols if c in df.columns]
df = df[present].rename(columns={
    "created_date_x":"company_created_date",
    "created_date_y":"deal_created_date"
})

# Sort: customers first, then opportunities, then by created date
df = df.sort_values(
    ["is_customer","is_opportunity","company_created_date","deal_created_date"],
    ascending=[False, False, True, True],
    na_position="last"
)

df.to_csv(OUT, index=False)
print("Wrote", OUT, df.shape)
