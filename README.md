# AMR Watch Nigeria 

A web-based antimicrobial resistance surveillance dashboard built for Nigerian health facilities. It aggregates antibiotic susceptibility testing results and surfaces resistance trends by organism, drug class, facility, and patient demographic in real time.

**Live demo:(https://amr-watch-nigeria.streamlit.app) 

---

## Why this exists

Medical laboratory scientists generate antibiotic susceptibility testing results every day across thousands of Nigerian health facilities. Those results reach the requesting doctor and stop there. It lacks aggregation and trend tracking.

When a doctor prescribes empirically before MCS results return, that decision runs on habit, convention, and whatever informal knowledge they've built up at that facility over time. If resistance to a drug has been climbing for three months, there's no system to tell them. They find out the hard way, or they don't find out at all.

Nigeria ranks 20th globally for age-standardised AMR mortality. Drug-resistant infections were associated with 263,400 deaths in 2019. The country launched its first national AMR survey only in late 2025. At the facility level, where prescribing decisions happen every hour, nothing exists to see what's coming.

AMR Watch Nigeria is built to change that.

---

## What it does

The dashboard gives clinicians, pharmacists, and infection control officers a facility-level resistance picture they can act on:

- **Antibiogram** — resistance profile for any organism, ranked by antibiotic, colour-coded by severity
- **Resistance trends** — track how resistance to a specific organism-antibiotic pair has moved over time, by month or quarter, with optional facility comparison
- **Demographics** — resistance rates by age group and sex for any organism-antibiotic combination
- **Geography** — MRSA, ESBL, and carbapenem resistance rates broken down by geopolitical zone, facility type, or individual facility
- **Raw data export** — download isolates or AST results as CSV for offline analysis

All views respond to sidebar filters: year, facility type, geopolitical zone, specific facility, specimen type, and ward.

---

## Current status

**Prototype — concept validation stage.**

The dashboard currently runs on synthetic data calibrated to NCDC AMR surveillance reports and published Nigerian clinical literature:

- MRSA prevalence: ~80% (NCDC sentinel surveillance 2019-2021)
- ESBL-producing Enterobacteriaceae: 60-80%
- Carbapenem-resistant Enterobacteriaceae: 20-30%

12 public tertiary facilities are represented across Nigeria's six geopolitical zones. Real facility data pipeline under development.

**Do not use for clinical or policy decisions.**

---

## Tech stack

- **Data processing:** Python, Pandas, NumPy
- **Visualisation:** Plotly
- **Dashboard:** Streamlit
- **Deployment:** Streamlit Cloud

Production stack (in development): Flask backend, Supabase database, Google Stitch frontend.

---

## Run locally

```bash
git clone https://github.com/0xobaa/amr-watch-nigeria
cd amr-watch-nigeria
pip install -r requirements.txt
streamlit run amr_dashboard.py
```

---

## Background

This project grew out of a cross-sectional study on *S. aureus* in post-operative wound infections across two hospitals in Ilorin, Nigeria, published in the UMYU Journal of Microbiology Research in 2023 (doi: 10.47430/ujmr.2381.013). The study found a 15.2% prevalence of *S. aureus* in surgical site infections, with 40% classified as MRSA.

The frustration that produced this dashboard: the data existed in the lab. It never reached the people writing the prescriptions.

---

## Author

**Abimbola Nurudeen Oba**
Medical Laboratory Scientist | Data Scientist
Abuja, Nigeria

hello@abimbolaoba.com · [GitHub: 0xobaa](https://github.com/0xobaa) · [@bimmzzzz](https://twitter.com/bimmzzzz)

---

*Prototype built as part of AMR Watch Nigeria — a health surveillance venture currently at the concept validation stage. Applications submitted to GCYLP 2026 (ITU/Huawei) and iDICE Founders Lab (Bank of Industry/AfDB).*
