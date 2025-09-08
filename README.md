# Stedi GTM Data Pipeline

A comprehensive data pipeline for generating realistic sales and marketing data for a healthcare API platform (Stedi). This project simulates the complete customer journey from company discovery through deal closure and billing, incorporating AI-powered enrichment and realistic business logic.

## Overview

This pipeline processes company data through multiple stages:
1. **Company Enrichment** - AI-powered ICP classification and technology stack analysis
2. **Deal Generation** - Realistic sales pipeline simulation with probability-based outcomes
3. **Billing Creation** - Revenue modeling for closed-won deals
4. **Data Unification** - Final dataset combining all business entities

## Features

- **AI-Powered Enrichment**: Uses OpenAI with web search to classify Ideal Customer Profiles (ICP) and analyze technology stacks
- **Realistic Sales Modeling**: Multi-factor probability models for deal creation and win rates
- **Healthcare-Specific Logic**: Specialized for healthcare data exchange (X12, FHIR, clearinghouse technologies)
- **Reproducible Results**: Seeded random generation for consistent outputs
- **Caching System**: Efficient API usage with local caching of enrichment results

## Project Structure

```
gtm-company/
├── config/                    # Configuration files
│   ├── icp_rules.yml         # ICP classification rules
│   └── previous_customers.yml # Known customer database
├── data/
│   ├── input/                # Source data
│   │   └── companies_clay.csv
│   ├── raw/                  # Intermediate outputs
│   │   ├── companies.csv
│   │   ├── deals.csv
│   │   └── billing.csv
│   ├── interim/              # Cache and temporary files
│   │   └── enrich_cache/     # AI enrichment cache
│   └── final/                # Final output
│       └── unified.csv
├── enrich/                   # AI enrichment modules
│   ├── icp_openai.py        # ICP classification
│   └── techstack_openai.py  # Technology stack analysis
├── src/                      # Main pipeline scripts
│   ├── 01_build_companies.py
│   ├── 02_build_deals.py
│   ├── 03_build_billing.py
│   └── 04_build_unified.py
└── docs/
    └── deals_generation_approach.md
```

## Prerequisites

### Python Dependencies

Install the required Python packages:

```bash
pip install pandas numpy pyyaml tldextract python-dateutil openai python-dotenv
```

### Environment Setup

1. Create a `.env` file in the project root with your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

2. Ensure you have the input data file at `data/input/companies_clay.csv` with the following columns:
   - `Name` - Company name
   - `Domain` - Company domain
   - `Description` - Company description
   - `Primary Industry` - Industry classification
   - `Size` - Employee count range
   - `Type` - Company type
   - `Location` - Geographic location
   - `Country` - Country code
   - `LinkedIn URL` - LinkedIn profile URL

## Usage

### Running the Complete Pipeline

Execute the scripts in sequence to run the full data pipeline:

```bash
# Step 1: Build and enrich companies
python src/01_build_companies.py

# Step 2: Generate sales deals
python src/02_build_deals.py

# Step 3: Create billing records
python src/03_build_billing.py

# Step 4: Unify all data
python src/04_build_unified.py
```

### Individual Scripts

#### 1. Company Enrichment (`01_build_companies.py`)
- Processes input company data
- Performs AI-powered ICP classification
- Analyzes technology stacks using web search
- Caches results for efficiency
- Outputs: `data/raw/companies.csv`

#### 2. Deal Generation (`02_build_deals.py`)
- Generates realistic sales deals based on company characteristics
- Uses multi-factor probability models
- Simulates sales pipeline stages and outcomes
- Outputs: `data/raw/deals.csv`

#### 3. Billing Creation (`03_build_billing.py`)
- Creates billing records for closed-won deals
- Models ARR based on company size and fit
- Outputs: `data/raw/billing.csv`

#### 4. Data Unification (`04_build_unified.py`)
- Combines all datasets into final analysis-ready format
- Calculates derived metrics (sales cycle, outcomes)
- Outputs: `data/final/unified.csv`

## Configuration

### ICP Rules (`config/icp_rules.yml`)
Configure Ideal Customer Profile criteria:
- Target industries
- Company size bands
- Required keywords
- Geographic restrictions

### Previous Customers (`config/previous_customers.yml`)
Maintain database of existing customers for:
- Higher deal probability
- Reduced sales cycle duration
- Improved win rates

## Key Features

### AI-Powered Enrichment
- **ICP Classification**: Determines if companies fit the ideal customer profile
- **Technology Stack Analysis**: Identifies healthcare technologies (X12, FHIR, clearinghouse)
- **Web Search Integration**: Uses OpenAI's web search for real-time company intelligence
- **Confidence Scoring**: Provides confidence levels for all classifications

### Realistic Sales Modeling
- **Multi-Factor Probability**: Considers ICP fit, technology stack, company size, and previous customer status
- **Stage Progression**: Simulates realistic sales pipeline movement
- **Leakage Modeling**: Accounts for deals lost at different stages
- **Duration Variability**: Realistic sales cycle lengths based on company characteristics

### Healthcare-Specific Logic
- **X12/Clearinghouse Boost**: Higher probability for companies using standard healthcare data formats
- **FHIR/HL7 Recognition**: Moderate boost for modern healthcare APIs
- **Industry Targeting**: Focus on healthcare, insurance, and health IT companies

## Output Data

### Final Dataset (`data/final/unified.csv`)
Contains comprehensive view of:
- **Company Information**: Name, domain, industry, size, location
- **Enrichment Data**: ICP classification, technology stack, confidence scores
- **Sales Pipeline**: Deal stages, dates, outcomes, sales cycle duration
- **Revenue Data**: ARR, MRR, billing terms for customers
- **Outcome Classification**: won/lost/open/no_opportunity

## Performance Notes

- **Caching**: Enrichment results are cached to minimize API calls
- **Reproducibility**: All random generation uses fixed seeds for consistent results
- **Scalability**: Pipeline handles thousands of companies efficiently
- **API Efficiency**: Web search results are cached to reduce OpenAI API costs

## Troubleshooting

### Common Issues

1. **Missing OpenAI API Key**: Ensure `.env` file contains valid `OPENAI_API_KEY`
2. **Input Data Format**: Verify `companies_clay.csv` has required columns
3. **Permission Errors**: Ensure write permissions for output directories
4. **API Rate Limits**: Caching system helps, but large datasets may require rate limiting

### Data Quality

- **Domain Validation**: Invalid domains are handled gracefully
- **Missing Data**: Scripts include fallback logic for incomplete records
- **Error Handling**: Individual company processing errors don't stop the pipeline

## Contributing

When modifying the pipeline:
1. Maintain the seeded random generation for reproducibility
2. Update configuration files for new ICP criteria
3. Test with small datasets before full runs
4. Document any changes to the probability models

## License

This project is for internal use and contains proprietary business logic for Stedi's go-to-market modeling.
