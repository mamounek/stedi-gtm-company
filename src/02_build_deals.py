import pandas as pd
import numpy as np
import uuid
from typing import Tuple, Dict, Any

# Configuration
RNG_SEED = 43
rng = np.random.default_rng(RNG_SEED)

def generate_id() -> str:
    # Generate unique identifier for deals
    return str(uuid.uuid4())

# Constants
SALES_OWNERS = [
    'Marissa', 'Maria', 'Henry', 'Isabella', 'Jack', 'Katherine', 'Luke', 'Mary',
    'Noah', 'Olivia', 'Peter', 'Quinn', 'Rachel', 'Samuel', 'Taylor', 'William', 'Zachary'
]

# Band multipliers for duration and win probability
DURATION_BY_BAND = {
    '2-10': 0.90, '11-50': 1.00, '51-200': 1.10, '201-500': 1.18,
    '501-1000': 1.22, '1001-5000': 1.30, '5001-10000': 1.38, '10001+': 1.45
}

WIN_MULT_BY_BAND = {
    '2-10': 1.00, '11-50': 1.00, '51-200': 1.00, '201-500': 0.97,
    '501-1000': 0.95, '1001-5000': 0.92, '5001-10000': 0.90, '10001+': 0.88
}

# Probability multipliers
ICP_CREATE_MULT = 0.30
ICP_WIN_MULT = 0.40
X12_CLR_MULT = 1.20
FHIR_MULT = 1.10
X12_CLR_WIN_MULT = 1.25
FHIR_WIN_MULT = 1.10

# Baseline probabilities
PREV_CUSTOMER_CREATE_MIN = 0.85
PREV_CUSTOMER_WIN_MIN = 0.70

# Duration adjustments
PREV_CUSTOMER_DURATION_MULT = 0.85
EVAL_MULT_X12_CLR = 1.10
NEGO_MULT_X12_CLR = 1.15
NEGO_MULT_FHIR = 1.05

# Load data
companies = pd.read_csv('data/raw/companies.csv', parse_dates=['created_date'])

def get_band_multiplier(band_label: str, multiplier_dict: Dict[str, float]) -> float:
    # Get multiplier for a given band label
    return float(multiplier_dict.get(str(band_label), 1.0))

def parse_health_tech(health_tech_str: str) -> Tuple[bool, bool, bool]:
    # Parse health technology flags from string
    tech_str = str(health_tech_str or '').lower()
    
    x12_codes = ['x12', '270', '271', '276', '277', '278', '835', '837']
    has_x12 = any(code in tech_str for code in x12_codes)
    
    has_fhir = 'fhir' in tech_str or 'hl7' in tech_str
    
    clearinghouse_terms = ['clearinghouse', 'availity', 'edifecs', 'change healthcare', 'optum']
    has_clearinghouse = any(term in tech_str for term in clearinghouse_terms)
    
    return has_x12, has_fhir, has_clearinghouse

def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    # Clamp value between min and max
    return max(min_val, min(max_val, value))

def generate_baseline_probability(alpha: float, beta: float) -> float:
    # Generate baseline probability using beta distribution
    return float(rng.beta(alpha, beta))

def add_jitter(sigma: float = 0.15) -> float:
    # Add random jitter using log-normal distribution
    return float(np.exp(rng.normal(0.0, sigma)))

def calculate_create_probability(icp: bool, icp_confidence: float, prev_customer: bool, 
                                has_x12: bool, has_fhir: bool, has_clearinghouse: bool, 
                                stack_confidence: float) -> float:
    # Calculate probability of creating a deal
    baseline = generate_baseline_probability(1.6, 3.0) * add_jitter(0.12)
    multiplier = 1.0
    
    # ICP boost
    if icp:
        multiplier *= (1.0 + ICP_CREATE_MULT * float(icp_confidence or 0.0))
    
    # Health tech boost
    if has_x12 or has_clearinghouse:
        multiplier *= X12_CLR_MULT
    elif has_fhir:
        multiplier *= FHIR_MULT
    
    # Stack confidence scaling
    multiplier *= (0.6 + 0.4 * float(stack_confidence or 0.0))
    
    probability = baseline * multiplier
    
    # Previous customer minimum
    if prev_customer:
        probability = max(probability, PREV_CUSTOMER_CREATE_MIN)
    
    return clamp(probability, 0.03, 0.97)

def calculate_win_probability(icp: bool, icp_confidence: float, prev_customer: bool,
                             has_x12: bool, has_fhir: bool, has_clearinghouse: bool,
                             stack_confidence: float, band_label: str) -> float:
    # Calculate probability of winning a deal
    baseline = generate_baseline_probability(1.5, 4.0) * add_jitter(0.12)
    multiplier = 1.0
    
    # ICP boost
    if icp:
        multiplier *= (1.0 + ICP_WIN_MULT * float(icp_confidence or 0.0))
    
    # Health tech boost
    if has_x12 or has_clearinghouse:
        multiplier *= X12_CLR_WIN_MULT
    elif has_fhir:
        multiplier *= FHIR_WIN_MULT
    
    # Band-specific win multiplier
    multiplier *= get_band_multiplier(band_label, WIN_MULT_BY_BAND)
    
    # Stack confidence scaling
    multiplier *= (0.6 + 0.4 * float(stack_confidence or 0.0))
    
    probability = baseline * multiplier
    
    # Previous customer minimum
    if prev_customer:
        probability = max(probability, PREV_CUSTOMER_WIN_MIN)
    
    return clamp(probability, 0.02, 0.98)

def calculate_duration_scales(band_label: str, prev_customer: bool, 
                            has_x12_or_clearinghouse: bool, has_fhir: bool) -> Tuple[float, float, float]:
    # Calculate duration scaling factors for different deal stages
    base_scale = get_band_multiplier(band_label, DURATION_BY_BAND)
    
    # Previous customer adjustment
    if prev_customer:
        base_scale *= PREV_CUSTOMER_DURATION_MULT
    
    # Evaluation multiplier
    eval_mult = EVAL_MULT_X12_CLR if (has_x12_or_clearinghouse or has_fhir) else 1.00
    
    # Negotiation multiplier
    if has_x12_or_clearinghouse:
        nego_mult = NEGO_MULT_X12_CLR
    elif has_fhir:
        nego_mult = NEGO_MULT_FHIR
    else:
        nego_mult = 1.00
    
    # Add jitter for realism
    return (base_scale * add_jitter(0.08), 
            eval_mult * add_jitter(0.06), 
            nego_mult * add_jitter(0.06))

def calculate_leakage_rates(icp: bool = False, prev_customer: bool = False, 
                          has_x12_or_clearinghouse: bool = False) -> Tuple[float, float, float]:
    # Calculate leakage rates for different deal stages
    base_eval = float(rng.beta(2.5, 20.0))  # ~0.11 average
    base_prop = float(rng.beta(2.5, 18.0))  # ~0.12 average
    base_neg = float(rng.beta(2.5, 18.0))   # ~0.12 average
    
    adjustment = 1.0
    if icp:
        adjustment *= 0.85
    if prev_customer:
        adjustment *= 0.70
    if has_x12_or_clearinghouse:
        adjustment *= 0.90
    
    return (clamp(base_eval * adjustment, 0.02, 0.25),
            clamp(base_prop * adjustment, 0.03, 0.30),
            clamp(base_neg * adjustment, 0.03, 0.30))

def process_company_deal(company_row: pd.Series) -> Dict[str, Any]:
    # Process a single company and generate deal data if applicable
    # Extract company attributes
    icp = bool(company_row.get('icp', False))
    icp_confidence = float(company_row.get('icp_confidence', 0.0) or 0.0)
    prev_customer = bool(company_row.get('is_prev_stedi_customer', False))
    stack_confidence = float(company_row.get('tech_stack_confidence', 0.0) or 0.0)
    band_label = company_row.get('employee_band', '')
    
    # Parse health technology flags
    has_x12, has_fhir, has_clearinghouse = parse_health_tech(company_row.get('health_tech', ''))
    has_x12_or_clearinghouse = has_x12 or has_clearinghouse
    
    # Calculate probabilities
    create_prob = calculate_create_probability(icp, icp_confidence, prev_customer, 
                                             has_x12, has_fhir, has_clearinghouse, stack_confidence)
    win_prob = calculate_win_probability(icp, icp_confidence, prev_customer,
                                       has_x12, has_fhir, has_clearinghouse, 
                                       stack_confidence, band_label)
    
    # Skip if no deal created
    if rng.random() >= create_prob:
        return None
    
    # Generate deal
    deal_id = generate_id()
    owner = rng.choice(SALES_OWNERS)
    
    # Calculate deal creation date
    created_date = company_row['created_date'] + pd.Timedelta(days=int(rng.normal(10, 7)))
    created_date = max(created_date, company_row['created_date'])
    
    # Calculate stage durations
    base_scale, eval_mult, nego_mult = calculate_duration_scales(
        band_label, prev_customer, has_x12_or_clearinghouse, has_fhir)
    
    discovery_days = max(1, int(rng.lognormal(mean=1.2, sigma=0.6) * base_scale))
    evaluation_days = max(1, int(rng.lognormal(mean=1.4, sigma=0.6) * base_scale * eval_mult))
    proposal_days = max(1, int(rng.lognormal(mean=1.3, sigma=0.6) * base_scale))
    negotiation_days = max(1, int(rng.lognormal(mean=1.3, sigma=0.6) * base_scale * nego_mult))
    
    # Calculate stage dates
    discovery_date = created_date + pd.Timedelta(days=discovery_days)
    evaluation_date = discovery_date + pd.Timedelta(days=evaluation_days)
    proposal_date = evaluation_date + pd.Timedelta(days=proposal_days)
    negotiation_date = proposal_date + pd.Timedelta(days=negotiation_days)
    
    # Calculate leakage rates and determine outcome
    leak_eval_rate, leak_prop_rate, leak_neg_rate = calculate_leakage_rates(
        icp=icp, prev_customer=prev_customer, has_x12_or_clearinghouse=has_x12_or_clearinghouse)
    
    closed_won_date = pd.NaT
    closed_lost_date = pd.NaT
    
    # Determine deal outcome based on leakage
    if rng.random() < leak_eval_rate:
        # Leaked during evaluation
        closed_lost_date = discovery_date + pd.Timedelta(days=int(rng.integers(1, 5)))
        evaluation_date = proposal_date = negotiation_date = pd.NaT
    elif rng.random() < leak_prop_rate:
        # Leaked during proposal
        closed_lost_date = evaluation_date + pd.Timedelta(days=int(rng.integers(1, 7)))
        proposal_date = negotiation_date = pd.NaT
    elif rng.random() < leak_neg_rate:
        # Leaked during negotiation
        closed_lost_date = proposal_date + pd.Timedelta(days=int(rng.integers(1, 10)))
        negotiation_date = pd.NaT
    else:
        # Deal completed - determine win/loss
        if rng.random() < win_prob:
            # Won
            close_days = max(1, int(rng.lognormal(mean=1.1, sigma=0.5) * base_scale))
            if prev_customer:
                close_days = max(1, int(close_days * 0.8))
            closed_won_date = negotiation_date + pd.Timedelta(days=close_days)
        else:
            # Lost
            lose_days = max(1, int(rng.lognormal(mean=1.2, sigma=0.6) * base_scale))
            closed_lost_date = negotiation_date + pd.Timedelta(days=lose_days)
    
    return {
        'deal_id': deal_id,
        'account_id': company_row['account_id'],
        'owner': owner,
        'created_date': created_date,
        'stage_date_discovery': discovery_date,
        'stage_date_evaluation': evaluation_date,
        'stage_date_proposal': proposal_date,
        'stage_date_negotiation': negotiation_date,
        'stage_date_closed_won': closed_won_date,
        'stage_date_closed_lost': closed_lost_date
    }

def main():
    # Main function to process companies and generate deals
    # Process all companies
    deals_data = []
    for _, company_row in companies.iterrows():
        deal = process_company_deal(company_row)
        if deal:
            deals_data.append(deal)
    
    # Create DataFrame and save
    deals_df = pd.DataFrame(deals_data)
    deals_df.to_csv('data/raw/deals.csv', index=False)
    print(f'Wrote data/raw/deals.csv with shape {deals_df.shape}')
    return deals_df

if __name__ == "__main__":
    deals = main()
