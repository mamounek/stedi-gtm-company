import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timezone

rng = np.random.default_rng(44)  # reproducible

# Helper functions
def uid(): return str(uuid.uuid4())

def parse_health_flags(health_tech_str: str):
    s = str(health_tech_str or '').lower()
    has_x12 = any(k in s for k in ['x12','270','271','276','277','278','835','837'])
    has_fhir = ('fhir' in s) or ('hl7' in s)
    has_clr  = any(k in s for k in ['clearinghouse','availity','edifecs','change healthcare','optum'])
    return has_x12, has_fhir, has_clr

def clamp(x, lo, hi): return max(lo, min(hi, x))

def arr_range_for_band(band: str):
    # Tight, strictly increasing ARR ranges so size dominates ARR
    table = {
        '2-10':        (6000,   14000),
        '11-50':       (12000,  28000),
        '51-200':      (28000,  80000),
        '201-500':     (60000,  140000),
        '501-1000':    (90000,  210000),
        '1001-5000':   (150000, 320000),
        '5001-10000':  (240000, 560000),
        '10001+':      (340000, 880000),
    }
    return table.get(str(band), (20000, 50000))

def round_currency(x, q=100):
    return int(np.round(x / q) * q)

def draw_arr_for_company(band: str, icp: bool, icp_conf: float, prev_cust: bool,
                         has_x12: bool, has_fhir: bool, has_clr: bool, stack_conf: float) -> int:
    # Draw ARR within band range with fit-based uplift
    # Size is primary driver, with small uplift for ICP/tech fit
    lo, hi = arr_range_for_band(band)

    # Size-first draw: centered but variable
    u = float(rng.beta(2.2, 2.2))          # 0..1 inside band
    base = lo + u * (hi - lo)

    # Fit score (0..1)
    rails = 1.0 if (has_x12 or has_clr) else (0.5 if has_fhir else 0.0)
    s = 0.0
    if icp: s += 0.45 * float(icp_conf or 0.0)
    s += 0.35 * rails
    if prev_cust: s += 0.15
    s += 0.05 * float(stack_conf or 0.0)
    s = float(clamp(s, 0.0, 1.0))

    uplift = 1.0 + (0.25 * s)
    uplift *= float(np.exp(rng.normal(0, 0.06)))  # tiny jitter

    arr = base * uplift

    # Guardrails (keep most within band; allow strong fits up to +30%)
    lo_guard = lo * 0.95
    hi_guard = hi * (1.0 + 0.30 * s)
    arr = clamp(arr, lo_guard, hi_guard)

    return round_currency(arr)

# Load inputs
companies = pd.read_csv('data/raw/companies.csv', parse_dates=['created_date'])
deals = pd.read_csv(
    'data/raw/deals.csv',
    parse_dates=[
        'created_date','stage_date_discovery','stage_date_evaluation',
        'stage_date_proposal','stage_date_negotiation',
        'stage_date_closed_won','stage_date_closed_lost'
    ]
)

# Only closed won deals become billing rows
won = deals[~deals['stage_date_closed_won'].isna()].copy()
if won.empty:
    print("No Closed Won deals found — writing empty billing.csv.")
    pd.DataFrame(columns=[]).to_csv('data/raw/billing.csv', index=False)
    raise SystemExit(0)

# Minimal company fields needed to size ARR
cols = [
    'account_id','employee_band','icp','icp_confidence','is_prev_stedi_customer',
    'health_tech','tech_stack_confidence'
]
acc = companies[cols].drop_duplicates('account_id')
won = won.merge(acc, on='account_id', how='left')

# Build simple billing (all annual, 12-month term)
ANNUAL_TERM_MONTHS = 12
BILLING_FREQUENCY  = 'annual'

rows = []
for _, r in won.iterrows():
    band        = r.get('employee_band', '11-50')
    icp         = bool(r.get('icp', False))
    icp_conf    = float(r.get('icp_confidence', 0.0) or 0.0)
    prev_cust   = bool(r.get('is_prev_stedi_customer', False))
    stack_conf  = float(r.get('tech_stack_confidence', 0.0) or 0.0)
    has_x12, has_fhir, has_clr = parse_health_flags(r.get('health_tech',''))

    arr = draw_arr_for_company(band, icp, icp_conf, prev_cust, has_x12, has_fhir, has_clr, stack_conf)
    mrr = round_currency(arr / 12.0)

    # Start near closed won (+0..30 days)
    cw = pd.Timestamp(r['stage_date_closed_won'])
    if cw.tz is None:
        cw = cw.tz_localize('UTC')
    else:
        cw = cw.tz_convert('UTC')
    start = (cw + pd.Timedelta(days=int(rng.integers(0, 31)))).normalize()
    end   = (start + pd.offsets.DateOffset(months=ANNUAL_TERM_MONTHS)).normalize()

    rows.append({
        'billing_id': uid(),
        'deal_id': r['deal_id'],
        'account_id': r['account_id'],
        'employee_band': band,
        'billing_start_date': start,
        'billing_end_date': end,
        'term_months': ANNUAL_TERM_MONTHS,
        'billing_frequency': BILLING_FREQUENCY,  # Always 'annual'
        'arr': arr,
        'mrr': mrr,  # Derived for reporting convenience
        'currency': 'USD'
    })

billing = pd.DataFrame(rows)
billing.to_csv('data/raw/billing.csv', index=False)
print("Wrote data/raw/billing.csv", billing.shape)
