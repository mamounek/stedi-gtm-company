# Build companies with enrichment
import os, json, uuid
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import tldextract
from dateutil import tz
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from enrich.techstack_openai import infer_tech_stack_via_websearch, score_stack_confidence
from enrich.icp_openai import decide_icp_via_websearch

# Paths and configuration
ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV  = ROOT / "data" / "input" / "companies_clay.csv"
OUTPUT_CSV = ROOT / "data" / "raw" / "companies.csv"             # <- same location, same filename family
CACHE_DIR  = ROOT / "data" / "interim" / "enrich_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RULES_YAML = ROOT / "config" / "icp_rules.yml"
PREV_YAML  = ROOT / "config" / "previous_customers.yml"
RULES = yaml.safe_load(RULES_YAML.read_text())
PREV  = yaml.safe_load(PREV_YAML.read_text())
PREV_DOMAINS = {str(c["domain"]).lower() for c in PREV.get("customers", [])}

# Helper functions
rng = np.random.default_rng(42)
def uid() -> str: return str(uuid.uuid4())

def band(n):
    if pd.isna(n): return rng.choice(['2-10','11-50','51-200','201-500','501-1000','1001-5000','5001-10000','10001+'], p=[.25,.25,.2,.1,.1,.05,.025,.025])
    return n.split()[0]

start = pd.Timestamp('2024-09-01', tz='UTC')
end   = pd.Timestamp('2025-08-31', tz='UTC')
def rand_date():
    return pd.to_datetime(rng.integers(start.value//10**9, end.value//10**9, endpoint=True), unit='s', utc=True)

SOURCES = ['paid_search','paid_social','content','referral','outbound','events','direct']
SRC_P   =  [0.22,         0.12,         0.08,      0.20,      0.25,      0.05,    0.08]
CAMPAIGNS = {
  'paid_search': ['SEM-Brand-US','SEM-Competitor','SEM-NonBrand'],
  'paid_social': ['PS-LinkedIn-MM','PS-Meta-SMB'],
  'content':     ['Ebook-AI-101','Webinar-EDI-Modern'],
  'referral':    ['Customer-Referral','Partner-Intro'],
  'outbound':    ['OB-Prospecting','OB-ABM-T1'],
  'events':      ['Conf-HIMSS','Conf-RevenueSummit'],
  'direct':      ['Direct-None']
}

def normalize_domain(d: str) -> str:
    d = (d or '').strip().lower()
    if not d: return ''
    ext = tldextract.extract(d)
    if not ext.domain: return d
    root = ".".join([ext.domain, ext.suffix]) if ext.suffix else ext.domain
    return root

def cache_file(domain: str) -> Path:
    return CACHE_DIR / f"{domain or 'no-domain'}.json"

# Main function
def main():
    df = pd.read_csv(INPUT_CSV)

    # Rename and map fields
    df = df.rename(columns={
        'Name':'account_name',
        'Domain':'account_domain',
        'Description':'account_description',
        'Primary Industry':'industry',
        'Size':'employee_count',    
        'Type':'type',
        'Country':'country', 
        'Location':'city',
        'LinkedIn URL':'linkedin_url',   
    })
    # Build base company structure
    out = pd.DataFrame()
    out['account_id']     = [uid() for _ in range(len(df))]                                    
    out['account_name']   = df.get('account_name', 'Unknown Co')                               
    out['account_domain'] = df.get('account_domain', pd.Series(['']*len(df)))                  
    out['description'] = df.get('account_description', pd.Series(['']*len(df)))        
    out['industry']       = df.get('industry', pd.Series(['Unknown']*len(df)))                 
    out['country']        = df.get('country', pd.Series(['US']*len(df)))                       
    out['employee_band']  = df.get('employee_count', np.nan).apply(band)                       
    out['created_date']   = [rand_date() for _ in range(len(df))]                              
    out['linkedin_url']   = df.get('linkedin_url', pd.Series(['']*len(df)))                    

    srcs = rng.choice(SOURCES, size=len(df), p=SRC_P)                                         
    out['source']   = srcs                                                                    
    out['campaign'] = [rng.choice(CAMPAIGNS[s]) for s in srcs]                                

    # Enrichment with web search and LLM
    apis_cloud, health_tech, patterns = [], [], []
    stack_conf, prev_flags = [], []
    icp_vals, icp_reasons, icp_confs = [], [], []

    for i, row in out.iterrows():
        name   = str(row['account_name'])
        domain = normalize_domain(str(row['account_domain']))
        band_l = str(row['employee_band'])
        country= str(row['country'])
        desc   = str(df.iloc[i].get('description', ''))  # from Clay input
        prev   = domain in PREV_DOMAINS
        prev_flags.append(prev)

        cpath = cache_file(domain)
        cached = json.loads(cpath.read_text()) if cpath.exists() else None

        if cached:
            stack = cached.get("stack", {})
            icp   = cached.get("icp", {})
        else:
            if not domain:
                stack = {"apis_cloud": [], "health_tech": [], "integration_patterns": [], "evidence": [], "tech_stack_confidence": 0.0}
                icp   = {"is_icp": False, "icp_reason": "No domain", "icp_confidence": 0.2, "matched_use_cases":[]}
            else:
                # Get tech stack
                try:
                    stack = infer_tech_stack_via_websearch(name, domain, band_l, country)
                except Exception:
                    stack = {"apis_cloud": [], "health_tech": [], "integration_patterns": [], "evidence": [], "tech_stack_confidence": 0.0}
                # Boost confidence score
                stack["tech_stack_confidence"] = score_stack_confidence(stack.get("tech_stack_confidence", 0.0), stack.get("evidence", []))

                # Determine ICP status
                comp_payload = {
                    "account_name": name,
                    "account_domain": domain,
                    "description": desc,
                    "employee_band": band_l,
                    "country": country
                }
                try:
                    icp = decide_icp_via_websearch(comp_payload, stack, prev, RULES)
                except Exception:
                    icp = {"is_icp": prev, "icp_reason": "Previous Stedi customer" if prev else "Insufficient evidence",
                           "icp_confidence": 0.9 if prev else 0.3, "matched_use_cases":[]}

            # Save to cache
            try:
                cpath.write_text(json.dumps({"stack": stack, "icp": icp}, ensure_ascii=False, indent=2))
            except Exception:
                pass

        apis_cloud.append("|".join(stack.get("apis_cloud", [])))
        health_tech.append("|".join(stack.get("health_tech", [])))
        patterns.append("|".join(stack.get("integration_patterns", [])))
        stack_conf.append(stack.get("tech_stack_confidence", 0.0))
        icp_vals.append(bool(icp.get("is_icp", False)))
        icp_reasons.append(str(icp.get("icp_reason", ""))[:140])
        icp_confs.append(float(icp.get("icp_confidence", 0.0)))

    # Add enrichment columns
    out['apis_cloud']             = apis_cloud
    out['health_tech']            = health_tech
    out['integration_patterns']   = patterns
    out['tech_stack_confidence']  = stack_conf
    out['is_prev_stedi_customer'] = prev_flags
    out['icp']                    = icp_vals
    out['icp_reason']             = icp_reasons
    out['icp_confidence']         = icp_confs

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}  rows={len(out)}  cols={len(out.columns)}")

if __name__ == "__main__":
    main()
