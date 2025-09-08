import os
import json
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

_client = OpenAI()

TECH_STACK_SCHEMA = {
    "name": "TechStackSchema",
    "schema": {
        "type": "object",
        "properties": {
            "apis_cloud": {"type": "array", "items": {"type": "string"}},          # e.g., ["AWS S3","Snowflake","Kafka","dbt"]
            "health_tech": {"type": "array", "items": {"type": "string"}},         # e.g., ["X12 270/271","FHIR R4","Epic","Availity"]
            "integration_patterns": {"type": "array", "items": {"type": "string"}},# e.g., ["API-first","SFTP EDI","VAN"]
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "maxLength": 160},
                        "url": {"type": "string"}
                    },
                    "required": ["claim", "url"],
                    "additionalProperties": False
                }
            },
            "tech_stack_confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["apis_cloud", "health_tech", "integration_patterns", "evidence", "tech_stack_confidence"],
        "additionalProperties": False
    },
    "strict": True
}

def infer_tech_stack_via_websearch(
    company_name: str,
    domain: str,
    employee_band: str,
    country: str
) -> Dict:
    """
    Uses OpenAI Responses API with Web Search to infer APIs/cloud + health-tech stack.
    Returns a dict conforming to TECH_STACK_SCHEMA.
    """
    system = (
        "You are an OSINT-style analyst. Use the web_search tool to examine the company's own "
        "developer/docs/engineering/careers/security/case-study pages and credible third-party docs. "
        "Identify ONLY technical APIs and infrastructure (cloud services, data platforms, streaming, "
        "integration middleware) and health-tech rails (HIPAA X12, specific transaction sets, FHIR/HL7, EHRs, "
        "clearinghouses). Exclude CMS/analytics pixels and generic marketing tools. Return JSON per schema. "
        "Include 2–6 evidence URLs with short claims."
    )
    user = (
        f"Company: {company_name}  Domain: {domain}  Country: {country}  Size band: {employee_band}\n"
        "Prioritize queries like:\n"
        f"- site:{domain} developer OR docs OR api\n"
        f"- site:{domain} security OR HIPAA\n"
        f"- site:{domain} careers engineer OR platform\n"
        "- 'X12' OR 'EDI' OR 'FHIR' OR 'HL7' on company pages\n"
        "- 'AWS' OR 'Azure' OR 'GCP' OR 'Snowflake' OR 'Kafka' OR 'dbt' OR 'Airflow' on company pages\n"
        "Return arrays apis_cloud[], health_tech[], integration_patterns[], evidence[], and tech_stack_confidence (0-1)."
    )
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                 {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": TECH_STACK_SCHEMA}
    )
    return json.loads(resp.choices[0].message.content)  # Parse JSON string to dict

def score_stack_confidence(model_conf: float, evidence: List[Dict]) -> float:
    """
    Deterministic booster based on evidence quality + count.
    docs/dev/security > careers > blog/case > everything else
    """
    if not isinstance(evidence, list):
        evidence = []
    weight = 0.0
    for e in evidence:
        u = str(e.get("url", "")).lower()
        if any(k in u for k in ["/docs", "developer.", "/developers", "/api", "/security", "/hipaa"]):
            weight += 0.25
        elif any(k in u for k in ["/careers", "/jobs"]):
            weight += 0.15
        elif any(k in u for k in ["/blog", "/engineering", "/case"]):
            weight += 0.10
        else:
            weight += 0.05
    weight = min(weight, 0.8)
    count_boost = min(len(evidence) * 0.03, 0.2)
    final = max(0.05, min(1.0, 0.5 * float(model_conf or 0) + 0.5 * (weight + count_boost)))
    return round(final, 2)
