import json
from typing import Dict
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

_client = OpenAI()

ICP_SCHEMA = {
    "name": "IcpDecision",
    "schema": {
        "type": "object",
        "properties": {
            "is_icp": {"type": "boolean"},
            "icp_reason": {"type": "string", "maxLength": 140},
            "icp_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "matched_use_cases": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["is_icp", "icp_reason", "icp_confidence", "matched_use_cases"],
        "additionalProperties": False
    },
    "strict": True
}

def decide_icp_via_websearch(
    company: Dict,              # {account_name, account_domain, description, employee_band, country}
    stack: Dict,                # from techstack_openai
    prev_customer: bool,        # domain in your previous_customers list
    rules: Dict                 # loaded YAML: industries_icp, size_bands_icp, countries_allowed, etc.
) -> Dict:
    """
    Uses OpenAI Web Search + context to decide ICP for Stedi-like use cases.
    Returns: {is_icp, icp_reason, icp_confidence, matched_use_cases[]}
    """
    name = company["account_name"]
    domain = company["account_domain"]
    desc = (company.get("description") or "")[:800]
    band = company.get("employee_band", "")
    country = company.get("country", "")

    system = (
        "You classify ICP for an API-first healthcare data exchange platform (clearinghouse + HIPAA X12). "
        "ICP are organizations likely to send/receive X12 (e.g., 270/271, 276/277, 278, 837/835), value API automation, "
        "and operate at target size/geos. Use web_search to verify public evidence on the company's site or credible sources. "
        "Be conservative if unclear. Return JSON per schema."
    )
    user = (
        f"Company: {name}  Domain: {domain}  Country: {country}  Size band: {band}\n"
        f"Description: {desc}\n"
        f"Previous customer: {prev_customer}\n"
        f"Tech stack: apis_cloud={stack.get('apis_cloud',[])}, health_tech={stack.get('health_tech',[])}, "
        f"integration_patterns={stack.get('integration_patterns',[])}, tech_stack_confidence={stack.get('tech_stack_confidence',0)}\n\n"
        "Search specifically for Stedi-style use cases on this company's public pages:\n"
        f"- site:{domain} X12 OR 270 OR 271 OR 276 OR 277 OR 278 OR 837 OR 835 OR clearinghouse\n"
        f"- site:{domain} edi OR payer OR provider OR remittance OR eligibility OR claims\n"
        "Decide: is_icp (bool), icp_reason (<=140 chars), icp_confidence (0–1), matched_use_cases[]."
    )

    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                 {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": ICP_SCHEMA}
    )
    out = json.loads(resp.choices[0].message.content)

    # deterministic post-processing
    if prev_customer:
        out["is_icp"] = True
        out["icp_confidence"] = max(float(out.get("icp_confidence", 0.0)), 0.9)
        out["icp_reason"] = "Previous Stedi customer"

    strong_codes = {"270", "271", "276", "277", "278", "835", "837", "x12"}
    mc = " ".join(out.get("matched_use_cases", [])).lower()
    if (company.get("country") in rules.get("countries_allowed", [])
        and company.get("employee_band") in rules.get("size_bands_icp", [])
        and any(code in mc for code in strong_codes)):
        out["icp_confidence"] = max(float(out.get("icp_confidence", 0.0)), 0.8)

    # coerce & cap
    out["is_icp"] = bool(out["is_icp"])
    out["icp_reason"] = str(out["icp_reason"])[:140]
    out["icp_confidence"] = float(out["icp_confidence"])
    return out

