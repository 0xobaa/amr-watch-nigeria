# HealthScope

A facility-level antimicrobial resistance surveillance dashboard for Nigerian health facilities. It pulls antibiotic susceptibility testing results together and shows resistance by organism, drug class, facility, and patient demographic, updated as new results come in.

**Live demo:** https://amr-watch-nigeria.streamlit.app

## Why this exists

Medical laboratory scientists produce antibiotic susceptibility results every day across thousands of Nigerian facilities. Each result reaches the requesting doctor and stops there. Nobody aggregates it. Nobody tracks the trend.

So when a doctor prescribes empirically, before the MCS result comes back, that call rests on habit, local convention, and whatever informal sense they've picked up at that facility. If resistance to a drug has been climbing for three straight months, no system tells them. They find out the hard way, or not at all.

Nigeria ranks 20th globally for age-standardised AMR mortality. Drug-resistant infections were linked to 263,400 deaths in 2019. The first national AMR survey only launched in late 2025. At the facility level, where prescriptions get written every hour, there's still nothing that shows what's coming.

HealthScope is built to close that gap.

## What it does

The dashboard gives clinicians, pharmacists, and infection-control officers a resistance picture they can act on:

- **Antibiogram** — resistance profile for any organism, ranked by antibiotic and colour-coded by severity
- **Resistance trends** — how resistance for an organism-antibiotic pair has moved over time, by month or quarter, with optional facility comparison
- **Demographics** — resistance rates by age group and sex for any organism-antibiotic combination
- **Geography** — MRSA, ESBL, and carbapenem resistance broken down by geopolitical zone, facility type, or individual facility
- **Raw export** — download isolates or AST results as CSV for offline work

Every view responds to the sidebar filters: year, facility type, geopolitical zone, specific facility, specimen type, and ward.

## Current status

Prototype, at the concept-validation stage.

The dashboard runs on synthetic data calibrated to NCDC AMR surveillance reports and published Nigerian clinical literature:

- MRSA prevalence around 80% (NCDC sentinel surveillance, 2019-2021)
- ESBL-producing Enterobacteriaceae, 60-80%
- Carbapenem-resistant Enterobacteriaceae, 20-30%

Twelve public tertiary facilities are represented across Nigeria's six geopolitical zones. The real facility-data pipeline is still in development.

**Do not use for clinical or policy decisions.**

## Tech stack

- Data processing: Python, pandas, NumPy
- Visualisation: Plotly
- Dashboard: Streamlit
- Deployment: Streamlit Cloud

Production stack in development: Flask backend, Supabase database, Google Stitch frontend.

## Run locally

```bash
git clone https://github.com/0xobaa/healthscope
cd healthscope
pip install -r requirements.txt
streamlit run amr_dashboard.py
```

## Background

This grew out of a cross-sectional study on *S. aureus* in post-operative wound infections across two hospitals in Ilorin, Nigeria, published in the UMYU Journal of Microbiology Research in 2023 (doi: 10.47430/ujmr.2381.013). The study found 15.2% prevalence of *S. aureus* in surgical site infections, 40% of it MRSA.

The frustration behind the whole thing: the data already existed in the lab. It just never reached the people writing the prescriptions.

## Author

Abimbola Nurudeen Oba — Medical Laboratory Scientist and Data Scientist, Abuja, Nigeria.

hello@abimbolaoba.com · GitHub [@0xobaa](https://github.com/0xobaa) · X [@bimmzzzz](https://x.com/bimmzzzz)

Prototype built as part of HealthScope, a health-surveillance venture at the concept-validation stage. Applications submitted to GCYLP 2026 (ITU/Huawei) and iDICE Founders Lab (Bank of Industry/AfDB).
