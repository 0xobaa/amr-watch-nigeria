"""
Nigerian AMR Surveillance Dashboard — Prototype v2
Synthetic dataset calibrated to NCDC AMR reports and published Nigerian literature:
  - MRSA prevalence: ~80%
  - Carbapenem resistance (Enterobacterales): 20-30%
  - ESBL production: 60-80%

Schema (two linked tables, WHONET/GLASS-style):
  isolates:    one row per isolate with patient + facility + organism metadata
  ast_results: one row per isolate-antibiotic pair (long format)

References informing the parameterization:
  - NCDC AMR Surveillance Reports (2017-2022)
  - Iregbu et al., Afr J Lab Med
  - Olalekan et al., BMC Infectious Diseases
  - Nigeria National Action Plan on AMR
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from contextlib import contextmanager
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nigerian AMR Surveillance Dashboard",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.set_page_config(
    page_title="AMR Watch Nigeria",
    page_icon="🧫",
    layout="centered"
)

st.caption(
    "For the best experience, view on a desktop or laptop browser. "
    "Mobile optimisation is coming in the next version."
)

# ---------------------------------------------------------------------------
# Reference registries
# ---------------------------------------------------------------------------
FACILITIES = {
    "LUTH (Lagos)":         {"state": "Lagos",   "zone": "South West",    "type": "Federal Teaching Hospital",   "weight": 0.13},
    "UCH (Ibadan)":         {"state": "Oyo",     "zone": "South West",    "type": "Federal Teaching Hospital",   "weight": 0.11},
    "ABUTH (Zaria)":        {"state": "Kaduna",  "zone": "North West",    "type": "Federal Teaching Hospital",   "weight": 0.09},
    "AKTH (Kano)":          {"state": "Kano",    "zone": "North West",    "type": "Federal Teaching Hospital",   "weight": 0.10},
    "UNTH (Enugu)":         {"state": "Enugu",   "zone": "South East",    "type": "Federal Teaching Hospital",   "weight": 0.08},
    "UPTH (Port Harcourt)": {"state": "Rivers",  "zone": "South South",   "type": "Federal Teaching Hospital",   "weight": 0.08},
    "UBTH (Benin)":         {"state": "Edo",     "zone": "South South",   "type": "Federal Teaching Hospital",   "weight": 0.07},
    "JUTH (Jos)":           {"state": "Plateau", "zone": "North Central", "type": "Federal Teaching Hospital",   "weight": 0.06},
    "UATH (Gwagwalada)":    {"state": "FCT",     "zone": "North Central", "type": "Federal Teaching Hospital",   "weight": 0.07},
    "NHA (Abuja)":          {"state": "FCT",     "zone": "North Central", "type": "Federal Specialist Hospital", "weight": 0.09},
    "LASUTH (Lagos)":       {"state": "Lagos",   "zone": "South West",    "type": "State Teaching Hospital",     "weight": 0.08},
    "FMC Owerri":           {"state": "Imo",     "zone": "South East",    "type": "Federal Medical Centre",      "weight": 0.04},
}

ORGANISMS = {
    # Weights scaled slightly to accommodate N. meningitidis. Total sums to 1.0.
    "Escherichia coli":          0.2329,
    "Klebsiella pneumoniae":     0.1875,
    "Staphylococcus aureus":     0.169,
    "Pseudomonas aeruginosa":    0.0937,
    "Acinetobacter baumannii":   0.0656,
    "Enterococcus faecalis":     0.0562,
    "Salmonella Typhi":          0.0468,
    "Streptococcus pneumoniae":  0.0375,
    "Proteus mirabilis":         0.0281,
    "Enterobacter cloacae":      0.0187,
    # HVS-relevant organisms. Candida has no AST panel.
    "Gardnerella vaginalis":     0.02,
    "Neisseria gonorrhoeae":     0.015,
    "Candida species":           0.015,
    # CSF / meningitis — primarily relevant to CSF specimens
    "Neisseria meningitidis":    0.014,
}

SPECIMENS = {
    "Urine": 0.32, "Blood": 0.22, "Wound swab": 0.18, "Sputum": 0.10,
    "Pus": 0.08, "CSF": 0.04, "Stool": 0.03, "High vaginal": 0.03,
}

WARDS = {
    "Medical ward": 0.22, "Surgical ward": 0.18, "ICU": 0.14, "Paediatrics": 0.13,
    "A&E": 0.12, "Obstetrics": 0.09, "Outpatient": 0.08, "Burns unit": 0.04,
}

CLINICAL_INDICATIONS = [
    "Suspected UTI",
    "Suspected pneumonia",
    "Suspected sepsis / bloodstream infection",
    "Suspected surgical site infection",
    "Suspected wound infection (non-surgical)",
    "Suspected skin and soft tissue infection",
    "Suspected abscess",
    "Suspected meningitis",
    "Suspected enteric fever",
    "Suspected gastroenteritis",
    "Suspected vaginal infection",
    "Suspected STI / PID",
    "Other / not specified",
]

# Conditional distribution of clinical indication given specimen type. Each list of
# (indication, weight) tuples must sum to 1.0. Indication is sampled *given* specimen
# rather than independently — a CSF sample is overwhelmingly drawn for suspected
# meningitis, a urine sample for suspected UTI, etc.
SPECIMEN_INDICATION_WEIGHTS = {
    "Urine": [
        ("Suspected UTI", 0.90),
        ("Suspected sepsis / bloodstream infection", 0.05),
        ("Other / not specified", 0.05),
    ],
    "Blood": [
        ("Suspected sepsis / bloodstream infection", 0.70),
        ("Suspected enteric fever", 0.20),
        ("Suspected meningitis", 0.05),
        ("Suspected pneumonia", 0.03),
        ("Other / not specified", 0.02),
    ],
    "Wound swab": [
        ("Suspected surgical site infection", 0.45),
        ("Suspected wound infection (non-surgical)", 0.45),
        ("Suspected skin and soft tissue infection", 0.08),
        ("Other / not specified", 0.02),
    ],
    "Sputum": [
        ("Suspected pneumonia", 0.85),
        ("Suspected abscess", 0.08),
        ("Suspected sepsis / bloodstream infection", 0.05),
        ("Other / not specified", 0.02),
    ],
    "Pus": [
        ("Suspected skin and soft tissue infection", 0.40),
        ("Suspected surgical site infection", 0.25),
        ("Suspected wound infection (non-surgical)", 0.20),
        ("Suspected abscess", 0.10),
        ("Suspected sepsis / bloodstream infection", 0.03),
        ("Other / not specified", 0.02),
    ],
    "CSF": [
        ("Suspected meningitis", 0.90),
        ("Suspected sepsis / bloodstream infection", 0.05),
        ("Other / not specified", 0.05),
    ],
    "Stool": [
        ("Suspected enteric fever", 0.65),
        ("Suspected gastroenteritis", 0.28),
        ("Suspected sepsis / bloodstream infection", 0.03),
        ("Other / not specified", 0.04),
    ],
    "High vaginal": [
        ("Suspected vaginal infection", 0.60),
        ("Suspected STI / PID", 0.37),
        ("Other / not specified", 0.03),
    ],
}

# Organism distribution conditioned on specimen type. Weights per specimen must sum
# to 1.0. This ensures clinically coherent pairings — Gardnerella and Candida only
# appear in HVS, Streptococcus pneumoniae dominates CSF, E. coli dominates urine, etc.
# Anchored to Nigerian published rates (NCDC surveillance, Iregbu et al., Olalekan et al.)
# and general clinical microbiology. Replace with measured rates when real data arrives.
SPECIMEN_ORGANISM_WEIGHTS = {
    "Urine": [
        ("Escherichia coli",         0.50),
        ("Klebsiella pneumoniae",     0.18),
        ("Enterococcus faecalis",     0.12),
        ("Proteus mirabilis",         0.08),
        ("Staphylococcus aureus",     0.05),
        ("Pseudomonas aeruginosa",    0.04),
        ("Enterobacter cloacae",      0.03),
    ],
    "Blood": [
        ("Staphylococcus aureus",     0.28),
        ("Klebsiella pneumoniae",     0.20),
        ("Escherichia coli",          0.18),
        ("Salmonella Typhi",          0.14),
        ("Streptococcus pneumoniae",  0.08),
        ("Pseudomonas aeruginosa",    0.06),
        ("Acinetobacter baumannii",   0.04),
        ("Enterobacter cloacae",      0.02),
    ],
    "Wound swab": [
        ("Staphylococcus aureus",     0.38),
        ("Escherichia coli",          0.20),
        ("Pseudomonas aeruginosa",    0.15),
        ("Klebsiella pneumoniae",     0.12),
        ("Proteus mirabilis",         0.08),
        ("Acinetobacter baumannii",   0.05),
        ("Enterococcus faecalis",     0.02),
    ],
    "Sputum": [
        ("Klebsiella pneumoniae",     0.30),
        ("Streptococcus pneumoniae",  0.25),
        ("Staphylococcus aureus",     0.18),
        ("Pseudomonas aeruginosa",    0.15),
        ("Acinetobacter baumannii",   0.08),
        ("Escherichia coli",          0.04),
    ],
    "Pus": [
        ("Staphylococcus aureus",     0.42),
        ("Escherichia coli",          0.18),
        ("Klebsiella pneumoniae",     0.14),
        ("Pseudomonas aeruginosa",    0.12),
        ("Proteus mirabilis",         0.08),
        ("Acinetobacter baumannii",   0.06),
    ],
    "CSF": [
        # S. pneumoniae dominant nationally; N. meningitidis more prominent
        # in the meningitis belt (North Central / North West zones).
        # These weights reflect a national average across all 12 sentinel sites.
        ("Streptococcus pneumoniae",  0.40),
        ("Neisseria meningitidis",    0.25),
        ("Klebsiella pneumoniae",     0.16),
        ("Escherichia coli",          0.12),
        ("Staphylococcus aureus",     0.07),
    ],
    "Stool": [
        ("Salmonella Typhi",          0.60),
        ("Escherichia coli",          0.25),
        ("Klebsiella pneumoniae",     0.10),
        ("Enterobacter cloacae",      0.05),
    ],
    "High vaginal": [
        ("Gardnerella vaginalis",     0.38),
        ("Candida species",           0.28),
        ("Neisseria gonorrhoeae",     0.16),
        ("Escherichia coli",          0.10),
        ("Staphylococcus aureus",     0.05),
        ("Enterococcus faecalis",     0.03),
    ],
}

ENTEROBACTERALES = ["Escherichia coli", "Klebsiella pneumoniae", "Proteus mirabilis",
                     "Enterobacter cloacae", "Salmonella Typhi"]

FACILITY_TYPE_MULT = {
    "Federal Teaching Hospital":   1.00,
    "Federal Specialist Hospital": 1.05,
    "State Teaching Hospital":     0.95,
    "Federal Medical Centre":      0.92,
}

WARD_MULT = {
    "ICU": 1.15, "Burns unit": 1.15, "Surgical ward": 1.05,
    "Medical ward": 1.00, "Paediatrics": 0.95, "A&E": 1.00,
    "Obstetrics": 0.95, "Outpatient": 0.85,
}

ORGANISM_ABBREV = {
    "Staphylococcus aureus":   "S. aureus",
    "Escherichia coli":        "E. coli",
    "Klebsiella pneumoniae":   "K. pneumoniae",
    "Pseudomonas aeruginosa":  "P. aeruginosa",
    "Acinetobacter baumannii": "A. baumannii",
    "Enterococcus faecalis":   "E. faecalis",
    "Streptococcus pneumoniae":"S. pneumoniae",
    "Enterobacter cloacae":    "E. cloacae",
    "Proteus mirabilis":       "P. mirabilis",
    "Salmonella Typhi":        "S. Typhi",
    "Gardnerella vaginalis":   "G. vaginalis",
    "Neisseria gonorrhoeae":   "N. gonorrhoeae",
    "Candida species":         "Candida spp.",
    "Neisseria meningitidis":  "N. meningitidis",
}

# ---------------------------------------------------------------------------
# Data generation — two linked tables
# ---------------------------------------------------------------------------
def _age_group(age: int) -> str:
    if age < 1:  return "<1 year"
    if age < 5:  return "1-4"
    if age < 15: return "5-14"
    if age < 45: return "15-44"
    if age < 65: return "45-64"
    return "65+"


def calculate_trend(ast_df: pd.DataFrame, organism: str, antibiotic: str,
                    min_periods: int = 2) -> str:
    """Compare resistance rate in the first vs second half of available quarters.

    Returns a direction string. Requires at least 20 tests total and at least
    min_periods quarters with n >= 10 to produce a meaningful result; otherwise
    returns '— Insufficient data' rather than a misleading direction.
    """
    pair_df = ast_df[
        (ast_df["organism"] == organism) &
        (ast_df["antibiotic"] == antibiotic)
    ].copy()

    if len(pair_df) < 20:
        return "— Insufficient data"

    pair_df["period"] = pd.to_datetime(pair_df["collection_date"]).dt.to_period("Q").astype(str)
    period_summary = (
        pair_df.groupby("period")
        .agg(n_tested=("interpretation", "count"),
             n_r=("interpretation", lambda x: (x == "R").sum()))
        .reset_index()
    )
    period_summary["n_tested"] = period_summary["n_tested"].astype(int)
    period_summary["n_r"] = period_summary["n_r"].astype(int)
    period_summary = period_summary[period_summary["n_tested"] >= 10].sort_values("period")

    if len(period_summary) < min_periods:
        return "— Insufficient data"

    period_summary["pct_r"] = period_summary["n_r"] / period_summary["n_tested"] * 100

    midpoint = len(period_summary) // 2
    first_half = period_summary.iloc[:midpoint]["pct_r"].mean()
    second_half = period_summary.iloc[midpoint:]["pct_r"].mean()
    diff = second_half - first_half

    if diff > 5:
        return "↑ Rising"
    elif diff < -5:
        return "↓ Falling"
    else:
        return "→ Stable"


def wilson_confidence_interval(n_r: int, n_tested: int, confidence: float = 0.95):
    """Wilson score interval for a proportion. Returns (lower, upper) on 0-100 scale.

    Wide when n is small, narrow when n is large — communicates reliability visually
    without forcing the user to interpret raw counts.
    """
    if n_tested == 0:
        return 0.0, 0.0
    p = n_r / n_tested
    z = stats.norm.ppf((1 + confidence) / 2)
    denominator = 1 + z ** 2 / n_tested
    center = (p + z ** 2 / (2 * n_tested)) / denominator
    margin = (z * (p * (1 - p) / n_tested + z ** 2 / (4 * n_tested ** 2)) ** 0.5) / denominator
    lower = max(0.0, (center - margin) * 100)
    upper = min(100.0, (center + margin) * 100)
    return round(lower, 1), round(upper, 1)


def _generate_ast_panel(org, context_mult, rng):
    """Return list of dicts: {antibiotic, antibiotic_class, interpretation, [_flag]}."""
    def draw(base):
        p = min(0.98, base * context_mult)
        r = rng.random()
        if r < p: return "R"
        if r < p + 0.04: return "I"
        return "S"

    P = []
    def add(ab, cls, interp, **extra):
        row = {"antibiotic": ab, "antibiotic_class": cls, "interpretation": interp}
        row.update(extra)
        P.append(row)

    if org == "Staphylococcus aureus":
        mrsa = rng.random() < min(0.98, 0.80 * context_mult)
        add("Oxacillin",          "Penicillins",        "R" if mrsa else "S", _mrsa=mrsa)
        add("Cefoxitin",          "Cephalosporins",     "R" if mrsa else "S")
        add("Vancomycin",         "Glycopeptides",      "R" if rng.random() < 0.02 else "S")
        add("Linezolid",          "Oxazolidinones",     "R" if rng.random() < 0.01 else "S")
        add("Clindamycin",        "Lincosamides",       draw(0.55 if mrsa else 0.20))
        add("Erythromycin",       "Macrolides",         draw(0.70 if mrsa else 0.30))
        add("Gentamicin",         "Aminoglycosides",    draw(0.50 if mrsa else 0.15))
        add("Trimethoprim-Sulfa", "Folate inhibitors",  draw(0.65 if mrsa else 0.35))
        add("Ciprofloxacin",      "Fluoroquinolones",   draw(0.70 if mrsa else 0.25))
        add("Tetracycline",       "Tetracyclines",      draw(0.60 if mrsa else 0.30))

    elif org in ENTEROBACTERALES:
        esbl_rates = {"Escherichia coli": 0.65, "Klebsiella pneumoniae": 0.75,
                      "Proteus mirabilis": 0.55, "Enterobacter cloacae": 0.60,
                      "Salmonella Typhi": 0.30}
        carb_rates = {"Escherichia coli": 0.18, "Klebsiella pneumoniae": 0.28,
                      "Proteus mirabilis": 0.15, "Enterobacter cloacae": 0.22,
                      "Salmonella Typhi": 0.05}
        esbl = rng.random() < min(0.95, esbl_rates[org] * context_mult)
        carb = rng.random() < min(0.90, carb_rates[org] * context_mult)

        add("Ampicillin",         "Penicillins",        draw(0.90), _esbl=esbl)
        add("Amoxicillin-Clav",   "Beta-lactam/BLI",    draw(0.70 if esbl else 0.35))
        add("Piperacillin-Tazo",  "Beta-lactam/BLI",    draw(0.45 if esbl else 0.15))
        add("Ceftriaxone",        "Cephalosporins",     "R" if esbl else draw(0.20))
        add("Ceftazidime",        "Cephalosporins",     "R" if esbl else draw(0.18))
        add("Cefepime",           "Cephalosporins",     "R" if esbl else draw(0.15))
        add("Meropenem",          "Carbapenems",        "R" if carb else "S", _carb=carb)
        add("Imipenem",           "Carbapenems",        "R" if carb else ("R" if rng.random() < 0.05 else "S"))
        add("Ertapenem",          "Carbapenems",        "R" if carb else ("R" if rng.random() < 0.08 else "S"))
        add("Ciprofloxacin",      "Fluoroquinolones",   draw(0.65 if esbl else 0.30))
        add("Gentamicin",         "Aminoglycosides",    draw(0.55 if esbl else 0.25))
        add("Amikacin",           "Aminoglycosides",    draw(0.25 if esbl else 0.10))
        add("Trimethoprim-Sulfa", "Folate inhibitors",  draw(0.75))

    elif org == "Pseudomonas aeruginosa":
        carb = rng.random() < min(0.85, 0.25 * context_mult)
        add("Meropenem",          "Carbapenems",        "R" if carb else "S", _carb=carb)
        add("Imipenem",           "Carbapenems",        "R" if carb else "S")
        add("Ceftazidime",        "Cephalosporins",     draw(0.35))
        add("Cefepime",           "Cephalosporins",     draw(0.30))
        add("Piperacillin-Tazo",  "Beta-lactam/BLI",    draw(0.30))
        add("Ciprofloxacin",      "Fluoroquinolones",   draw(0.40))
        add("Gentamicin",         "Aminoglycosides",    draw(0.45))
        add("Amikacin",           "Aminoglycosides",    draw(0.20))
        add("Colistin",           "Polymyxins",         "R" if rng.random() < 0.03 else "S")

    elif org == "Acinetobacter baumannii":
        carb = rng.random() < min(0.90, 0.60 * context_mult)
        add("Meropenem",          "Carbapenems",        "R" if carb else "S", _carb=carb)
        add("Imipenem",           "Carbapenems",        "R" if carb else "S")
        add("Ceftazidime",        "Cephalosporins",     draw(0.75))
        add("Cefepime",           "Cephalosporins",     draw(0.70))
        add("Ciprofloxacin",      "Fluoroquinolones",   draw(0.75))
        add("Gentamicin",         "Aminoglycosides",    draw(0.65))
        add("Amikacin",           "Aminoglycosides",    draw(0.45))
        add("Trimethoprim-Sulfa", "Folate inhibitors",  draw(0.80))
        add("Colistin",           "Polymyxins",         "R" if rng.random() < 0.05 else "S")

    elif org == "Enterococcus faecalis":
        add("Ampicillin",         "Penicillins",        draw(0.25))
        add("Vancomycin",         "Glycopeptides",      "R" if rng.random() < 0.05 else "S")
        add("Linezolid",          "Oxazolidinones",     "R" if rng.random() < 0.02 else "S")
        add("Gentamicin (high)",  "Aminoglycosides",    draw(0.55))
        add("Ciprofloxacin",      "Fluoroquinolones",   draw(0.60))
        add("Tetracycline",       "Tetracyclines",      draw(0.70))

    elif org == "Streptococcus pneumoniae":
        add("Penicillin",         "Penicillins",        draw(0.30))
        add("Ceftriaxone",        "Cephalosporins",     draw(0.15))
        add("Erythromycin",       "Macrolides",         draw(0.40))
        add("Trimethoprim-Sulfa", "Folate inhibitors",  draw(0.70))
        add("Vancomycin",         "Glycopeptides",      "S")
        add("Levofloxacin",       "Fluoroquinolones",   draw(0.10))

    elif org == "Neisseria gonorrhoeae":
        add("Ceftriaxone",        "Cephalosporins",     draw(0.15))
        add("Ciprofloxacin",      "Fluoroquinolones",   draw(0.55))
        add("Azithromycin",       "Macrolides",         draw(0.35))
        add("Penicillin",         "Penicillins",        draw(0.75))
        add("Spectinomycin",      "Aminocyclitols",     draw(0.05))
        add("Tetracycline",       "Tetracyclines",      draw(0.60))

    elif org == "Gardnerella vaginalis":
        add("Metronidazole",      "Nitroimidazoles",    draw(0.15))
        add("Clindamycin",        "Lincosamides",       draw(0.20))
        add("Ampicillin",          "Penicillins",       draw(0.25))
        add("Amoxicillin-Clav",   "Beta-lactam/BLI",    draw(0.15))

    elif org == "Neisseria meningitidis":
        # Ceftriaxone remains the drug of choice — resistance still very low in Nigeria.
        # Penicillin resistance rising but still below fluoroquinolone rates.
        # Chloramphenicol included — still used in resource-limited settings.
        add("Ceftriaxone",        "Cephalosporins",     draw(0.05))
        add("Penicillin",         "Penicillins",        draw(0.20))
        add("Ciprofloxacin",      "Fluoroquinolones",   draw(0.15))
        add("Chloramphenicol",    "Amphenicols",        draw(0.25))
        add("Rifampicin",         "Rifamycins",         draw(0.05))
        add("Azithromycin",       "Macrolides",         draw(0.10))

    # Candida species — no AST panel. Antifungal susceptibility is a separate workflow
    # and isn't in scope for this dataset. Falls through to `return P` with an empty list.

    return P


@st.cache_data
def generate_amr_dataset(n_isolates: int = 30000, seed: int = 42):
    rng = np.random.default_rng(seed)

    # Patient pool — some patients contribute multiple isolates (see weight comment below)
    n_patients = int(n_isolates * 0.82)

    # Sampling weights determine how likely each patient is to be picked for a new isolate.
    # Most patients have weight 1 (single isolate); a minority get higher weights so they
    # appear multiple times in the output — simulating repeat cultures from the same patient,
    # which is common in hospital surveillance. The [1,1,1,2,3]/[0.7,0.1,0.05,0.1,0.05]
    # mix yields roughly 45-50% of patients contributing >1 isolate after normalisation.
    pw = rng.choice([1, 1, 1, 2, 3], size=n_patients, p=[0.7, 0.1, 0.05, 0.1, 0.05]).astype(float)
    pw /= pw.sum()

    # Vectorise the per-patient attributes once, rather than per-isolate via a dict lookup
    patient_ages = np.clip(rng.normal(38, 22, size=n_patients), 0, 95).astype(int)
    patient_sexes = rng.choice(["Male", "Female"], size=n_patients, p=[0.48, 0.52])

    fac_names = list(FACILITIES.keys())
    fw = np.array([FACILITIES[f]["weight"] for f in fac_names])
    fw /= fw.sum()

    # --- Batch sampling: do all of these in one call each rather than n_isolates × 6 calls.
    # This is what keeps 30k generation fast (~3 seconds instead of ~2+ minutes).
    # Organism is NOT batch-sampled here — it is drawn per-isolate conditioned on the
    # specimen type so that organism-specimen pairings are clinically coherent.
    pid_idx = rng.choice(n_patients, size=n_isolates, p=pw)
    fac_idx = rng.choice(len(fac_names), size=n_isolates, p=fw)
    spec_keys = list(SPECIMENS.keys())
    spec_idx = rng.choice(len(spec_keys), size=n_isolates, p=list(SPECIMENS.values()))
    ward_keys = list(WARDS.keys())
    ward_idx = rng.choice(len(ward_keys), size=n_isolates, p=list(WARDS.values()))

    day_offsets = rng.integers(0, 1095, size=n_isolates)
    tat_days = rng.integers(2, 9, size=n_isolates)  # turnaround 2-8 days inclusive

    start = datetime(2023, 1, 1)

    isolates, ast_rows = [], []
    for i in range(n_isolates):
        pid_i = int(pid_idx[i])
        pid = f"PT-NG-{pid_i + 1:06d}"
        fac = fac_names[int(fac_idx[i])]
        fmeta = FACILITIES[fac]
        spec = spec_keys[int(spec_idx[i])]
        ward = ward_keys[int(ward_idx[i])]

        # Draw organism conditioned on specimen type — ensures clinically coherent pairings.
        # Same pattern as clinical_indication draw; per-isolate cost is negligible (~3μs).
        org_options = SPECIMEN_ORGANISM_WEIGHTS[spec]
        org_labels = [x[0] for x in org_options]
        org_probs = np.array([x[1] for x in org_options])
        org = str(rng.choice(org_labels, p=org_probs))

        # Draw clinical indication conditioned on specimen type — a urine sample is
        # overwhelmingly for suspected UTI, blood for sepsis, etc. Per-isolate rng.choice
        # on a list of 3-6 items is cheap (~3 microseconds) so the loop cost is negligible.
        ind_options = SPECIMEN_INDICATION_WEIGHTS[spec]
        ind_labels = [x[0] for x in ind_options]
        ind_probs = np.array([x[1] for x in ind_options])
        clinical_indication = str(rng.choice(ind_labels, p=ind_probs))

        cdate = start + timedelta(days=int(day_offsets[i]))
        rdate = cdate + timedelta(days=int(tat_days[i]))

        context_mult = (WARD_MULT[ward]
                         * FACILITY_TYPE_MULT[fmeta["type"]]
                         * (1 + 0.03 * (cdate.year - 2023)))

        iid = f"NG-{cdate.year}-{i+1:05d}"
        age = int(patient_ages[pid_i])
        row = {
            "isolate_id": iid,
            "patient_id": pid,
            "patient_age": age,
            "patient_age_group": _age_group(age),
            "patient_sex": str(patient_sexes[pid_i]),
            "collection_date": cdate,
            "result_date": rdate,
            "year": cdate.year,
            "month": cdate.strftime("%Y-%m"),
            "quarter": f"Q{((cdate.month - 1) // 3) + 1} {cdate.year}",
            "facility": fac,
            "facility_type": fmeta["type"],
            "state": fmeta["state"],
            "zone": fmeta["zone"],
            "ward_type": ward,
            "specimen_type": spec,
            "clinical_indication": clinical_indication,
            "organism": org,
            "MRSA_status": "N/A",
            "ESBL_status": "N/A",
            "carbapenem_resistant": "N/A",
        }

        panel = _generate_ast_panel(org, context_mult, rng)

        # Lift phenotype flags onto isolate row
        for r in panel:
            if "_mrsa" in r:
                row["MRSA_status"] = "MRSA" if r.pop("_mrsa") else "MSSA"
            if "_esbl" in r:
                row["ESBL_status"] = "Positive" if r.pop("_esbl") else "Negative"
            if "_carb" in r:
                row["carbapenem_resistant"] = "Yes" if r.pop("_carb") else "No"

        for r in panel:
            ast_rows.append({
                "isolate_id": iid,
                "organism": org,
                "antibiotic": r["antibiotic"],
                "antibiotic_class": r["antibiotic_class"],
                "interpretation": r["interpretation"],
                "method": "Disk diffusion",
            })

        isolates.append(row)

    isolates_df = pd.DataFrame(isolates)
    isolates_df["collection_date"] = pd.to_datetime(isolates_df["collection_date"])
    isolates_df["result_date"] = pd.to_datetime(isolates_df["result_date"])

    ast_df = pd.DataFrame(ast_rows)
    return isolates_df, ast_df


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
isolates_df, ast_df = generate_amr_dataset()


@st.cache_data
def ast_with_meta(_ast, _iso):
    return _ast.merge(
        _iso[["isolate_id", "facility", "facility_type", "state", "zone",
              "ward_type", "specimen_type", "clinical_indication",
              "collection_date", "year", "quarter",
              "patient_age_group", "patient_sex"]],
        on="isolate_id", how="left"
    )


ast_full = ast_with_meta(ast_df, isolates_df)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🧫 AMR Dashboard")
st.sidebar.caption("Nigerian Antimicrobial Resistance Surveillance — Prototype")

st.sidebar.subheader("Filters")
st.sidebar.caption("Leave a filter empty to include all values.")

# Primary filters — always visible for quick exploration
year_f = st.sidebar.multiselect("Year", sorted(isolates_df["year"].unique()), default=[])

zone_f = st.sidebar.multiselect("Geopolitical zone",
                                 sorted(isolates_df["zone"].unique()), default=[])

# Facility options cascade from the zone selection — if the user picks "South West",
# only LUTH, UCH and LASUTH appear in the facility dropdown. If no zone is selected,
# all facilities are available.
if zone_f:
    available_facilities = sorted(
        isolates_df[isolates_df["zone"].isin(zone_f)]["facility"].unique()
    )
else:
    available_facilities = sorted(isolates_df["facility"].unique())

fac_f = st.sidebar.multiselect("Facility", available_facilities, default=[])

clin_ind_f = st.sidebar.multiselect(
    "Clinical indication",
    sorted(isolates_df["clinical_indication"].unique()),
    default=[],
)

# Advanced filters — collapsed by default. Specimen and ward are power-user filters
# that often over-constrain a first-look exploration if left visible.
with st.sidebar.expander("Advanced filters", expanded=False):
    spec_f = st.multiselect("Specimen type",
                             sorted(isolates_df["specimen_type"].unique()), default=[])
    ward_f = st.multiselect("Ward type",
                             sorted(isolates_df["ward_type"].unique()), default=[])

# Facility type filter is hidden for v1 — all 12 facilities are public tertiary, so the
# filter has nothing to discriminate on. Will return when secondary / primary sites
# are added to the dataset. The column itself is preserved for geography and KPI logic.


def apply_filter(df: pd.DataFrame, column: str, selection: list) -> pd.DataFrame:
    """Return df unchanged if selection is empty; otherwise filter by isin(selection)."""
    if not selection:
        return df
    return df[df[column].isin(selection)]


iso_f = isolates_df
iso_f = apply_filter(iso_f, "year", year_f)
iso_f = apply_filter(iso_f, "zone", zone_f)
iso_f = apply_filter(iso_f, "facility", fac_f)
iso_f = apply_filter(iso_f, "clinical_indication", clin_ind_f)
iso_f = apply_filter(iso_f, "specimen_type", spec_f)
iso_f = apply_filter(iso_f, "ward_type", ward_f)

ast_f = ast_full[ast_full["isolate_id"].isin(iso_f["isolate_id"])]

iso_f["organism_display"] = iso_f["organism"].map(lambda x: ORGANISM_ABBREV.get(x, x))
ast_f["organism_display"] = ast_f["organism"].map(lambda x: ORGANISM_ABBREV.get(x, x))

st.sidebar.markdown("---")
st.sidebar.metric("Isolates in selection", f"{len(iso_f):,}")
st.sidebar.metric("AST results", f"{len(ast_f):,}")
st.sidebar.metric("Unique patients", f"{iso_f['patient_id'].nunique():,}")
st.sidebar.caption("Synthetic data. Prototype / development use only.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Nigerian AMR Surveillance Dashboard")
st.markdown("Prototype dashboard — AMR patterns across Nigerian public tertiary facilities.")

if len(iso_f) == 0:
    st.warning("No isolates match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Quick filters — above the tabs so they affect every tab
# Visible on mobile without opening the sidebar hamburger menu.
# They narrow the same iso_f / ast_f that every tab uses.
# ---------------------------------------------------------------------------
st.caption("Quick filters — applies to all tabs")

qf_row1_col1, qf_row1_col2 = st.columns(2)
with qf_row1_col1:
    quick_year = st.selectbox(
        "Year",
        options=["All"] + sorted(isolates_df["year"].unique().tolist(), reverse=True),
        key="quick_year",
    )
with qf_row1_col2:
    quick_facility = st.selectbox(
        "Facility",
        options=["All facilities"] + sorted(isolates_df["facility"].unique().tolist()),
        key="quick_facility",
    )

quick_indication = st.selectbox(
    "Clinical indication",
    options=["All indications"] + sorted(isolates_df["clinical_indication"].unique().tolist()),
    key="quick_indication",
)

st.caption("For specimen type, ward, and zone filters — tap the menu icon at the top left.")
st.divider()

# Apply quick filters on top of sidebar filters
if quick_year != "All":
    iso_f = iso_f[iso_f["year"] == int(quick_year)]
if quick_facility != "All facilities":
    iso_f = iso_f[iso_f["facility"] == quick_facility]
if quick_indication != "All indications":
    iso_f = iso_f[iso_f["clinical_indication"] == quick_indication]

# Rebuild ast_f and display columns after quick filters are applied
ast_f = ast_full[ast_full["isolate_id"].isin(iso_f["isolate_id"])]
iso_f["organism_display"] = iso_f["organism"].map(lambda x: ORGANISM_ABBREV.get(x, x))
ast_f["organism_display"] = ast_f["organism"].map(lambda x: ORGANISM_ABBREV.get(x, x))

# Dynamic default organism — the most prevalent in the current filtered selection.
# Used as the default selectbox value in Tabs 3, 4, and 5 so the first thing a
# user sees reflects the context they filtered for, not a hardcoded organism.
default_org = iso_f["organism"].value_counts().idxmax() if len(iso_f) > 0 else "Escherichia coli"

if len(iso_f) == 0:
    st.info(
        "No isolates match the current filter combination. "
        "Try broadening your Year, Facility, or Clinical indication selection, "
        "or clear the sidebar filters."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["📊 Overview", "🧬 Organisms", "💊 Antibiograms", "📈 Resistance trends",
     "👥 Demographics", "🗺️ Geography", "📋 Raw data"]
)


# Friendly empty-state handler. Wrapping each tab's content prevents one failure
# from taking down the whole dashboard. The Details expander still shows the raw
# error for debugging — this should never be hidden from the developer, only from
# the end user on first glance.
@contextmanager
def tab_guard():
    try:
        yield
    except Exception as exc:
        st.info(
            "Limited data for the current filter combination. Try broadening your "
            "selection — for example, choose a wider date range or include more facilities."
        )
        with st.expander("Technical details (for debugging)"):
            st.code(f"{type(exc).__name__}: {exc}")

# ---- Tab 1: Overview — focused "what should I worry about?" view ----
with tab1, tab_guard():

        # --- KPI cards ---
        c1, c2, c3, c4, c5 = st.columns(5)

        sa = iso_f[iso_f["organism"] == "Staphylococcus aureus"]
        mrsa_rate = (sa["MRSA_status"] == "MRSA").mean() * 100 if len(sa) else np.nan

        entero = iso_f[iso_f["organism"].isin(ENTEROBACTERALES)
                               & iso_f["ESBL_status"].isin(["Positive", "Negative"])]
        esbl_rate = (entero["ESBL_status"] == "Positive").mean() * 100 if len(entero) else np.nan

        carb_e = iso_f[iso_f["organism"].isin(ENTEROBACTERALES)
                               & iso_f["carbapenem_resistant"].isin(["Yes", "No"])]
        carb_rate = (carb_e["carbapenem_resistant"] == "Yes").mean() * 100 if len(carb_e) else np.nan

        c1.metric("Total isolates", f"{len(iso_f):,}")
        c2.metric("Unique patients", f"{iso_f['patient_id'].nunique():,}")
        c3.metric("MRSA prevalence", f"{mrsa_rate:.1f}%" if not np.isnan(mrsa_rate) else "—")
        c4.metric("ESBL prevalence", f"{esbl_rate:.1f}%" if not np.isnan(esbl_rate) else "—")
        c5.metric("Carbapenem-R (Entero.)", f"{carb_rate:.1f}%" if not np.isnan(carb_rate) else "—")

        st.markdown("---")

        # --- Resistance markers over time ---
        st.subheader("Resistance markers over time")
        st.caption("Tracks the three headline resistance markers across quarters for the "
                    "current filter selection.")

        t = iso_f.copy()
        t["period"] = t["collection_date"].dt.to_period("Q").astype(str)

        # Include n per period so we can hide low-volume periods where the rate would be
        # driven by 2-3 isolates rather than real signal.
        MIN_PERIOD_N = 20

        sa_t = t[t["organism"] == "Staphylococcus aureus"].groupby("period").apply(
            lambda g: pd.Series({
                "MRSA %": (g["MRSA_status"] == "MRSA").mean() * 100,
                "n": len(g),
            }),
            include_groups=False,
        ).reset_index()
        sa_t = sa_t[sa_t["n"] >= MIN_PERIOD_N]

        esbl_t = t[t["organism"].isin(ENTEROBACTERALES)
                    & t["ESBL_status"].isin(["Positive", "Negative"])].groupby("period").apply(
            lambda g: pd.Series({
                "ESBL %": (g["ESBL_status"] == "Positive").mean() * 100,
                "n": len(g),
            }),
            include_groups=False,
        ).reset_index()
        esbl_t = esbl_t[esbl_t["n"] >= MIN_PERIOD_N]

        carb_t = t[t["carbapenem_resistant"].isin(["Yes", "No"])
                    & t["organism"].isin(ENTEROBACTERALES)].groupby("period").apply(
            lambda g: pd.Series({
                "Carbapenem-R %": (g["carbapenem_resistant"] == "Yes").mean() * 100,
                "n": len(g),
            }),
            include_groups=False,
        ).reset_index()
        carb_t = carb_t[carb_t["n"] >= MIN_PERIOD_N]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sa_t["period"], y=sa_t["MRSA %"],
                                  mode="lines+markers", name="MRSA",
                                  customdata=sa_t["n"],
                                  hovertemplate="%{y:.1f}% (n=%{customdata})<extra>MRSA</extra>"))
        fig.add_trace(go.Scatter(x=esbl_t["period"], y=esbl_t["ESBL %"],
                                  mode="lines+markers", name="ESBL",
                                  customdata=esbl_t["n"],
                                  hovertemplate="%{y:.1f}% (n=%{customdata})<extra>ESBL</extra>"))
        fig.add_trace(go.Scatter(x=carb_t["period"], y=carb_t["Carbapenem-R %"],
                                  mode="lines+markers", name="Carbapenem-R",
                                  customdata=carb_t["n"],
                                  hovertemplate="%{y:.1f}% (n=%{customdata})<extra>Carbapenem-R</extra>"))
        fig.update_layout(xaxis_title="Quarter", yaxis_title="% resistant",
                           hovermode="x unified", height=440, yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Periods with fewer than {MIN_PERIOD_N} isolates are hidden to avoid noise.")

        st.markdown("---")
        st.subheader("Top resistance concerns")
        st.caption("The five highest-resistance organism × antibiotic pairs in the current "
                    "selection. Use this as a quick 'what to watch' list.")

        # Rank all organism × antibiotic pairs by %R. The threshold is 5 (rather than the
        # earlier 30) so that tight filter combinations still produce a useful panel; rows
        # with n < 10 are flagged as Low confidence so clinicians know to interpret cautiously.
        MIN_TESTS = 5
        concerns = (ast_f.groupby(["organism", "antibiotic", "antibiotic_class"])
                     .agg(n_tested=("interpretation", "count"),
                          n_r=("interpretation", lambda x: (x == "R").sum()))
                     .reset_index())
        # Cast to plain int to avoid PyArrow-backed dtype division errors on newer pandas
        concerns["n_tested"] = concerns["n_tested"].astype(int)
        concerns["n_r"] = concerns["n_r"].astype(int)
        concerns["pct_r"] = (concerns["n_r"] / concerns["n_tested"] * 100).round(1)

        # Exclude combinations with well-known intrinsic / near-universal resistance that
        # aren't clinically actionable — e.g. ampicillin in Gram-negatives is essentially
        # always resistant by nature, so clinicians don't use it empirically. Surfacing it
        # as a "top concern" adds noise to the signal.
        INTRINSIC_R_GRAM_NEG = ["Escherichia coli", "Klebsiella pneumoniae",
                                 "Proteus mirabilis", "Enterobacter cloacae",
                                 "Salmonella Typhi", "Pseudomonas aeruginosa",
                                 "Acinetobacter baumannii"]
        mask_intrinsic = ((concerns["organism"].isin(INTRINSIC_R_GRAM_NEG))
                           & (concerns["antibiotic"] == "Ampicillin"))
        concerns = concerns[~mask_intrinsic]

        concerns = concerns[concerns["n_tested"] >= MIN_TESTS].sort_values("pct_r", ascending=False).head(5)

        if len(concerns) == 0:
            st.info(
                "Limited data in the current filter combination to rank resistance concerns. "
                "Broaden filters (wider date range or more facilities) to see facility-wide "
                "resistance patterns."
            )
        else:
            concerns["organism_display"] = concerns["organism"].map(
                lambda x: ORGANISM_ABBREV.get(x, x)
            )
            concerns["label"] = concerns["organism_display"] + " / " + concerns["antibiotic"]
            fig = px.bar(concerns.sort_values("pct_r"), x="pct_r", y="label", orientation="h",
                          text="pct_r", color="pct_r", color_continuous_scale="Reds",
                          range_color=[0, 100], hover_data={"n_tested": True,
                                                             "antibiotic_class": True,
                                                             "label": False, "pct_r": False})
            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="inside",
                textfont=dict(size=11, color="white"),
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=220,
                xaxis=dict(showticklabels=False, title=""),
                yaxis=dict(tickfont=dict(size=11), title=""),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            display_concerns = concerns[["organism", "antibiotic", "pct_r", "n_tested"]].copy()
            display_concerns.columns = ["Organism", "Antibiotic", "% Resistant", "Isolates tested"]
            st.dataframe(display_concerns, use_container_width=True, hide_index=True)

# ---- Tab 2 ----
with tab2, tab_guard():
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Most isolated organisms")
        oc = iso_f["organism"].value_counts().reset_index()
        oc.columns = ["organism", "count"]
        oc["pct"] = (oc["count"] / oc["count"].sum() * 100).round(1)
        oc["organism_display"] = oc["organism"].map(lambda x: ORGANISM_ABBREV.get(x, x))
        fig = px.bar(oc, x="count", y="organism_display", orientation="h", text="pct")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(xaxis_title="Isolates", yaxis_title="", height=500,
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Counts")
        st.dataframe(oc[["organism", "count", "pct"]], hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Organism × specimen heatmap")

    # Abbreviate specimen names so x-axis labels stay readable on narrow screens
    specimen_abbrev = {
        "Blood": "Blood", "Urine": "Urine", "Wound swab": "Wound",
        "CSF": "CSF", "Sputum": "Sputum", "High vaginal": "HVS",
        "Pus": "Pus", "Stool": "Stool",
    }
    iso_heatmap = iso_f.copy()
    iso_heatmap["specimen_type"] = iso_heatmap["specimen_type"].map(
        lambda x: specimen_abbrev.get(x, x)
    )
    iso_heatmap["organism_display"] = iso_heatmap["organism"].map(
        lambda x: ORGANISM_ABBREV.get(x, x)
    )
    piv = pd.crosstab(iso_heatmap["organism_display"], iso_heatmap["specimen_type"])
    fig = px.imshow(piv, aspect="auto", color_continuous_scale="OrRd",
                     labels={"color": "Count"})
    fig.update_layout(
        xaxis=dict(tickangle=-30, tickfont=dict(size=10), title=""),
        yaxis=dict(tickfont=dict(size=10), title=""),
        margin=dict(l=100, r=20, t=20, b=60),
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 3: Antibiograms — the big win of long format ----
with tab3, tab_guard():
    st.subheader("Antibiogram — resistance profile by organism")
    st.caption(
        "Candida species are excluded from resistance analysis — antifungal "
        "susceptibility testing is not included in this dataset."
    )

    orgs = sorted(ast_f["organism"].unique())
    if not orgs:
        st.info("No AST data for current filter.")
    else:
        default_i = orgs.index(default_org) if default_org in orgs else 0
        sel = st.selectbox("Organism", orgs, index=default_i)

        oa = ast_f[ast_f["organism"] == sel]

        # Long format makes this a one-liner
        summary = (oa.groupby(["antibiotic", "antibiotic_class"])
                    .agg(n_tested=("interpretation", "count"),
                         n_r=("interpretation", lambda x: (x == "R").sum()))
                    .reset_index())
        summary["n_tested"] = summary["n_tested"].astype(int)
        summary["n_r"] = summary["n_r"].astype(int)
        summary["pct_r"] = (summary["n_r"] / summary["n_tested"] * 100).round(1)
        summary = summary[summary["n_tested"] >= 5].sort_values("pct_r")

        # Confidence label — informs clinicians which rates to trust
        summary["confidence"] = summary["n_tested"].apply(
            lambda n: "Low" if n < 10 else ("Medium" if n < 30 else "High")
        )

        if len(summary) == 0:
            st.info("Not enough tests per antibiotic to display an antibiogram for this "
                    "organism under the current filter. Try broadening the selection.")
        else:
            if (summary["n_tested"] < 10).any():
                st.caption("⚠️ Bars marked **Low** confidence have fewer than 10 tests. "
                            "Interpret with caution.")
            c1, c2 = st.columns([2, 1])
            with c1:
                fig = px.bar(summary, x="pct_r", y="antibiotic", orientation="h",
                              text="pct_r", color="pct_r",
                              color_continuous_scale="RdYlGn_r", range_color=[0, 100],
                              hover_data=["antibiotic_class", "n_tested", "confidence"])
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
                fig.update_layout(xaxis_title="% Resistant", yaxis_title="",
                                   height=max(400, 40 * len(summary)),
                                   xaxis_range=[0, 110], coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown(f"**{ORGANISM_ABBREV.get(sel, sel)}** — {oa['isolate_id'].nunique():,} isolates")
                disp = summary[["antibiotic", "antibiotic_class", "pct_r",
                                 "n_tested", "confidence"]].copy()
                disp.columns = ["Antibiotic", "Class", "% Resistant",
                                "Isolates tested", "Confidence"]
                st.dataframe(disp, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Resistance by antibiotic class (across all organisms in selection)")
    cs = (ast_f.groupby("antibiotic_class")
            .agg(n_tested=("interpretation", "count"),
                 n_r=("interpretation", lambda x: (x == "R").sum()))
            .reset_index())
    cs["n_tested"] = cs["n_tested"].astype(int)
    cs["n_r"] = cs["n_r"].astype(int)
    cs["pct_r"] = (cs["n_r"] / cs["n_tested"] * 100).round(1)
    cs = cs.sort_values("pct_r")
    fig = px.bar(cs, x="pct_r", y="antibiotic_class", orientation="h", text="pct_r",
                  color="pct_r", color_continuous_scale="RdYlGn_r", range_color=[0, 100])
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(xaxis_title="% Resistant", yaxis_title="", height=400,
                       xaxis_range=[0, 110], coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 4: Resistance trends — the early-warning clinical view ----
with tab4, tab_guard():
    st.subheader("Resistance trend — organism × antibiotic over time")
    st.caption("Track how resistance to a specific antibiotic has evolved for a given organism. "
                "Useful for detecting emerging resistance and informing empirical therapy. "
                "Candida species are excluded — antifungal susceptibility testing is not "
                "included in this dataset.")

    trend_orgs = sorted(ast_f["organism"].unique())
    if not trend_orgs:
        st.info("No AST data for current filter.")
    else:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            t_org = st.selectbox("Organism", trend_orgs,
                                  index=trend_orgs.index(default_org)
                                  if default_org in trend_orgs else 0,
                                  key="trend_org")

        # Antibiotics dynamically tied to the selected organism
        org_abs = sorted(ast_f[ast_f["organism"] == t_org]["antibiotic"].unique())
        with col_b:
            t_ab = st.selectbox("Antibiotic", org_abs,
                                 index=org_abs.index("Meropenem")
                                 if "Meropenem" in org_abs else 0,
                                 key="trend_ab")
        with col_c:
            period = st.radio("Period", ["Quarterly", "Monthly"], horizontal=True,
                              key="trend_period")

        # Optional facility comparison
        compare = st.checkbox("Compare across facilities", value=False)
        chosen_facs = None
        skip_chart = False
        if compare:
            # Default to the 3 highest-volume facilities in the current selection, so the
            # comparison starts with statistically meaningful signal rather than the 3 sites
            # that happen to come first alphabetically. Limiting to 3 keeps the chart
            # readable — more than that becomes spaghetti.
            vol_ranked = (ast_f[ast_f["organism"] == t_org]["facility"]
                           .value_counts().index.tolist())
            try:
                chosen_facs = st.multiselect(
                    "Facilities to compare (max 3)", vol_ranked,
                    default=vol_ranked[:3], max_selections=3,
                )
            except TypeError:
                # Older Streamlit versions don't support max_selections — fall back to
                # a soft limit with a warning.
                chosen_facs = st.multiselect(
                    "Facilities to compare (max 3)", vol_ranked,
                    default=vol_ranked[:3],
                )
                if len(chosen_facs) > 3:
                    st.warning("Please select a maximum of 3 facilities for a readable "
                                "comparison. Using the first 3.")
                    chosen_facs = chosen_facs[:3]

            if not chosen_facs:
                st.info("Select at least one facility to enable comparison, "
                        "or uncheck 'Compare across facilities' to see a single combined trend.")
                skip_chart = True

        # Build the trend dataset
        t_data = ast_f[(ast_f["organism"] == t_org) & (ast_f["antibiotic"] == t_ab)].copy()
        if compare and chosen_facs:
            t_data = t_data[t_data["facility"].isin(chosen_facs)]

        if skip_chart:
            pass  # user needs to pick facilities first
        elif len(t_data) == 0:
            st.warning("No records for this organism × antibiotic combination in the current filter.")
        else:
            if period == "Monthly":
                t_data["period"] = t_data["collection_date"].dt.to_period("M").astype(str)
            else:
                t_data["period"] = (
                    "Q" + t_data["collection_date"].dt.quarter.astype(str) +
                    " " + t_data["collection_date"].dt.year.astype(str)
                )

            group_cols = ["period"] + (["facility"] if compare else [])
            trend_summary = (t_data.groupby(group_cols)
                              .agg(n_tested=("interpretation", "count"),
                                   n_r=("interpretation", lambda x: (x == "R").sum()))
                              .reset_index())
            trend_summary["n_tested"] = trend_summary["n_tested"].astype(int)
            trend_summary["n_r"] = trend_summary["n_r"].astype(int)
            trend_summary["pct_r"] = (trend_summary["n_r"] / trend_summary["n_tested"] * 100).round(1)

            # Sort period chronologically BEFORE filtering, so rolling average preserves
            # chronological ordering.
            if period == "Monthly":
                trend_summary = trend_summary.sort_values("period")
            else:
                trend_summary["_sort"] = trend_summary["period"].apply(
                    lambda s: (int(s.split()[1]), int(s.split()[0][1:]))
                )
                trend_summary = trend_summary.sort_values("_sort").drop(columns="_sort")

            # --- Fix 7: minimum period volume filter ---
            # Hide periods based on fewer than 15 tests rather than show noisy rates
            # that would mislead interpretation. The message below tells the user how
            # many periods were hidden so they can broaden their filter if needed.
            MIN_VOLUME = 15
            low_volume_periods = trend_summary[trend_summary["n_tested"] < MIN_VOLUME]
            trend_summary = trend_summary[trend_summary["n_tested"] >= MIN_VOLUME].reset_index(drop=True)

            if len(trend_summary) == 0:
                st.info(
                    f"Not enough test volume to show reliable trends for {t_org} × {t_ab} "
                    f"under the current filters. Each period needs at least {MIN_VOLUME} tests. "
                    "Try broadening your date range, switching to quarterly, or including more facilities."
                )
            else:
                if len(low_volume_periods) > 0:
                    st.caption(
                        f"{len(low_volume_periods)} period(s) hidden due to insufficient "
                        f"test volume (n < {MIN_VOLUME})."
                    )

                # --- Fix 1: rolling average smoothing ---
                # In single-trace mode: smooth across the single chronological series.
                # In compare mode: smooth within each facility's own time series so the
                # rolling window doesn't mix facilities.
                if compare:
                    trend_summary["pct_r_smooth"] = (
                        trend_summary.groupby("facility")["pct_r"]
                        .rolling(window=2, min_periods=1).mean()
                        .round(1).reset_index(level=0, drop=True)
                    )
                else:
                    trend_summary["pct_r_smooth"] = (
                        trend_summary["pct_r"]
                        .rolling(window=2, min_periods=1).mean().round(1)
                    )

                # --- Fix 2: Wilson confidence interval (single-trace mode only) ---
                # In compare mode, overlapping CI bands would be unreadable, so we skip
                # CI visualisation there and let the smoothed per-facility lines speak.
                if not compare:
                    ci_vals = trend_summary.apply(
                        lambda row: pd.Series(
                            wilson_confidence_interval(row["n_r"], row["n_tested"])
                        ),
                        axis=1,
                    )
                    trend_summary[["ci_low", "ci_high"]] = ci_vals

                # --- Build the chart ---
                fig = go.Figure()

                if compare:
                    # One smoothed line per facility, raw points as faint markers
                    for fac, grp in trend_summary.groupby("facility"):
                        fig.add_trace(go.Scatter(
                            x=grp["period"], y=grp["pct_r_smooth"],
                            mode="lines+markers", line=dict(width=2.5),
                            name=fac,
                            customdata=grp["n_tested"],
                            hovertemplate=(f"<b>{fac}</b><br>%{{x}}<br>"
                                            "Smoothed: %{y:.1f}%<br>"
                                            "n=%{customdata}<extra></extra>"),
                        ))
                else:
                    # Single-trace: confidence band, raw scatter, smoothed main line
                    fig.add_trace(go.Scatter(
                        x=trend_summary["period"].tolist() +
                          trend_summary["period"].tolist()[::-1],
                        y=trend_summary["ci_high"].tolist() +
                          trend_summary["ci_low"].tolist()[::-1],
                        fill="toself",
                        fillcolor="rgba(99, 110, 250, 0.12)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="95% confidence band",
                        showlegend=True, hoverinfo="skip",
                    ))
                    fig.add_trace(go.Scatter(
                        x=trend_summary["period"], y=trend_summary["pct_r"],
                        mode="markers",
                        marker=dict(size=6, color="grey", opacity=0.5),
                        name="Raw per-period rate",
                        customdata=trend_summary["n_tested"],
                        hovertemplate="Raw: %{y:.1f}% (n=%{customdata})<extra></extra>",
                    ))
                    fig.add_trace(go.Scatter(
                        x=trend_summary["period"], y=trend_summary["pct_r_smooth"],
                        mode="lines", line=dict(width=3, color="rgb(99, 110, 250)"),
                        name=f"{t_ab} (smoothed)",
                        hovertemplate="Smoothed: %{y:.1f}%<extra></extra>",
                    ))

                # --- Fix 3: 50% clinical threshold line ---
                fig.add_hline(
                    y=50, line_dash="dash", line_color="rgba(220, 50, 50, 0.6)",
                    line_width=1.5,
                    annotation_text="50% threshold",
                    annotation_position="top right",
                    annotation_font_size=11,
                    annotation_font_color="rgba(220, 50, 50, 0.8)",
                )

                # --- Fix 4: latest period annotation (single-trace only to avoid clutter) ---
                if not compare:
                    latest_row = trend_summary.iloc[-1]
                    fig.add_annotation(
                        x=latest_row["period"],
                        y=latest_row["pct_r_smooth"],
                        text=f"  {latest_row['pct_r_smooth']:.1f}%",
                        showarrow=False,
                        font=dict(size=12, color="black"),
                        xanchor="left",
                    )

                # --- Fix 6: x-axis rotation and spacing ---
                display_org = ORGANISM_ABBREV.get(t_org, t_org)
                fig.update_layout(
                    xaxis=dict(
                        tickangle=-45,
                        tickfont=dict(size=9),
                        title="",
                    ),
                    yaxis_title=f"% Resistant to {t_ab}",
                    yaxis_range=[0, 100],
                    hovermode="x unified",
                    height=300,
                    title=dict(
                        text=f"{display_org} × {t_ab}",
                        font=dict(size=13),
                        x=0,
                        xanchor="left",
                    ),
                    margin=dict(l=50, r=15, t=40, b=80),
                    legend=dict(
                        font=dict(size=9),
                        orientation="v",
                        x=1.0, xanchor="right",
                        y=1.0, yanchor="top",
                        bgcolor="rgba(0,0,0,0)",
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- Summary metrics (Fix 8: trend direction) ---
                overall_r = (t_data["interpretation"] == "R").mean() * 100
                total_tested = len(t_data)
                period_label = "month" if period == "Monthly" else "quarter"

                # Latest-period %R — compute differently depending on compare mode
                if compare:
                    # Pool all facilities' tests in the latest period for a true weighted rate
                    latest_period = trend_summary["period"].iloc[-1]
                    latest_rows = trend_summary[trend_summary["period"] == latest_period]
                    latest_pct = (latest_rows["n_r"].sum() / latest_rows["n_tested"].sum() * 100) \
                                  if latest_rows["n_tested"].sum() > 0 else float("nan")
                    latest_label = f"Latest {period_label} %R (pooled)"
                    n_with_data = len(latest_rows)
                    n_selected = len(chosen_facs)
                    latest_help = (
                        f"Pooled rate across {n_with_data} of {n_selected} selected facilities "
                        f"for {latest_period} (sites with no tests in that period are excluded). "
                        f"Weighted by test volume, not a simple average of facility percentages."
                    )
                else:
                    latest_pct = trend_summary["pct_r"].iloc[-1]
                    latest_label = f"Latest {period_label} %R"
                    latest_help = f"Resistance rate for the most recent {period_label}."

                # Trend direction — compare first half vs second half of the filtered series.
                # Use raw pct_r (not smoothed) so real movement isn't dampened. In compare mode,
                # aggregate across facilities per period first so we compare the pooled trend.
                if compare:
                    pooled = (trend_summary.groupby("period", sort=False)
                              .apply(lambda g: (g["n_r"].sum() / g["n_tested"].sum() * 100),
                                     include_groups=False)
                              .reset_index(name="pct_r"))
                    trend_series = pooled["pct_r"]
                else:
                    trend_series = trend_summary["pct_r"]

                if len(trend_series) >= 3:
                    first_half = trend_series.iloc[:len(trend_series) // 2].mean()
                    second_half = trend_series.iloc[len(trend_series) // 2:].mean()
                    diff = second_half - first_half
                    if diff > 5:
                        trend_direction = "↑ Rising"
                    elif diff < -5:
                        trend_direction = "↓ Falling"
                    else:
                        trend_direction = "→ Stable"
                    trend_help = (f"Compares the first half of the visible period "
                                   f"({first_half:.1f}%) to the second half "
                                   f"({second_half:.1f}%). Rising / Falling is flagged at a "
                                   f"±5 percentage point change.")
                else:
                    trend_direction = "Insufficient data"
                    trend_help = "Need at least 3 periods to compute a trend direction."

                col_x, col_y, col_z, col_w = st.columns(4)
                col_x.metric("Overall %R", f"{overall_r:.1f}%")
                col_y.metric("Total tests", f"{total_tested:,}")
                col_z.metric(latest_label, f"{latest_pct:.1f}%", help=latest_help)
                col_w.metric("Trend", trend_direction, help=trend_help)

                with st.expander("Data table"):
                    show = trend_summary.copy()
                    # Explicit column rename — cleaner than blanket title-casing, which
                    # produced awkward labels like "N R" and "Pct R".
                    rename_map = {
                        "period": "Period",
                        "facility": "Facility",
                        "n_tested": "n tested",
                        "n_r": "n resistant",
                        "pct_r": "%R",
                        "pct_r_smooth": "%R (smoothed)",
                        "ci_low": "95% CI low",
                        "ci_high": "95% CI high",
                    }
                    # Drop any columns we don't want in the data table
                    show = show.rename(columns=rename_map)
                    st.dataframe(show, hide_index=True, use_container_width=True)

# ---- Tab 5: Demographics ----
with tab5, tab_guard():
    st.subheader("Resistance by patient demographics")
    st.caption("See whether resistance rates differ by patient age or sex for a chosen "
                "organism × antibiotic. Differences may reflect exposure patterns, "
                "prior antibiotic use, or community vs. hospital acquisition.")

    demo_orgs = sorted(ast_f["organism"].unique())
    if not demo_orgs:
        st.info("No AST data for current filter.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            d_org = st.selectbox("Organism", demo_orgs,
                                  index=demo_orgs.index(default_org)
                                  if default_org in demo_orgs else 0,
                                  key="demo_org")
        org_abs = sorted(ast_f[ast_f["organism"] == d_org]["antibiotic"].unique())
        with col_b:
            d_ab = st.selectbox("Antibiotic", org_abs,
                                 index=org_abs.index("Ciprofloxacin")
                                 if "Ciprofloxacin" in org_abs else 0,
                                 key="demo_ab")

        d_data = ast_f[(ast_f["organism"] == d_org) & (ast_f["antibiotic"] == d_ab)].copy()

        # Age group ordering for the chart below
        age_order = ["<1 year", "1-4", "5-14", "15-44", "45-64", "65+"]

        if len(d_data) == 0:
            st.warning("No records for this combination.")
        else:
            st.markdown("**Resistance by age group**")
            age_sum = (d_data.groupby("patient_age_group")
                        .agg(n=("interpretation", "count"),
                             n_r=("interpretation", lambda x: (x == "R").sum()))
                        .reset_index())
            age_sum["n"] = age_sum["n"].astype(int)
            age_sum["n_r"] = age_sum["n_r"].astype(int)
            age_sum["pct_r"] = (age_sum["n_r"] / age_sum["n"] * 100).round(1)
            # Drop age groups with fewer than 5 tests — too small to show any rate at all
            age_sum = age_sum[age_sum["n"] >= 5]
            age_sum["patient_age_group"] = pd.Categorical(
                age_sum["patient_age_group"], categories=age_order, ordered=True
            )
            age_sum = age_sum.sort_values("patient_age_group")
            age_sum["confidence"] = age_sum["n"].apply(
                lambda n: "Low" if n < 10 else ("Medium" if n < 30 else "High")
            )

            if len(age_sum) == 0:
                st.info("Not enough tests per age group to display a breakdown for this "
                        "combination under the current filter. Try broadening the selection.")
            else:
                fig = px.bar(age_sum, x="patient_age_group", y="pct_r", text="pct_r",
                              color="pct_r", color_continuous_scale="RdYlGn_r",
                              range_color=[0, 100], hover_data=["n", "confidence"])
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
                fig.update_layout(xaxis_title="Age group",
                                   yaxis_title=f"% Resistant to {d_ab}",
                                   yaxis_range=[0, 110], height=420,
                                   coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

                # Sample size warning — flag any age group with low test volume
                small_cells = age_sum[age_sum["n"] < 10]
                if len(small_cells) > 0:
                    st.warning(
                        f"⚠️ {len(small_cells)} age group(s) have fewer than 10 tests "
                        "(marked **Low** confidence). Interpret those rates with caution."
                    )

                with st.expander("Data table"):
                    show = age_sum[["patient_age_group", "n", "n_r", "pct_r",
                                     "confidence"]].copy()
                    show.columns = ["Age group", "Isolates tested", "Resistant",
                                     "% Resistant", "Confidence"]
                    st.dataframe(show, hide_index=True, use_container_width=True)

# ---- Tab 6: Geography ----
with tab6, tab_guard():
    st.subheader("Resistance by geography")
    marker = st.radio("Marker", ["MRSA", "ESBL", "Carbapenem-R (Enterobacterales)"],
                       horizontal=True)
    level = st.radio("Aggregate by", ["Zone", "Facility type", "Facility"], horizontal=True)
    group_col = {"Zone": "zone", "Facility type": "facility_type", "Facility": "facility"}[level]

    if marker == "MRSA":
        sub = iso_f[iso_f["organism"] == "Staphylococcus aureus"]
        rates = sub.groupby(group_col).apply(
            lambda g: pd.Series({"rate": (g["MRSA_status"] == "MRSA").mean() * 100, "n": len(g)}),
            include_groups=False
        ).reset_index()
    elif marker == "ESBL":
        sub = iso_f[iso_f["organism"].isin(ENTEROBACTERALES)
                     & iso_f["ESBL_status"].isin(["Positive", "Negative"])]
        rates = sub.groupby(group_col).apply(
            lambda g: pd.Series({"rate": (g["ESBL_status"] == "Positive").mean() * 100, "n": len(g)}),
            include_groups=False
        ).reset_index()
    else:
        sub = iso_f[iso_f["carbapenem_resistant"].isin(["Yes", "No"])
                      & iso_f["organism"].isin(ENTEROBACTERALES)]
        rates = sub.groupby(group_col).apply(
            lambda g: pd.Series({"rate": (g["carbapenem_resistant"] == "Yes").mean() * 100,
                                  "n": len(g)}),
            include_groups=False
        ).reset_index()

    fig = px.bar(rates.sort_values("rate"), x="rate", y=group_col, orientation="h",
                  text="rate", color="rate", color_continuous_scale="Reds",
                  range_color=[0, 100], hover_data=["n"])
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(xaxis_title=f"% {marker}", yaxis_title="", height=500,
                       xaxis_range=[0, 110], coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    disp = rates.copy()
    disp["confidence"] = disp["n"].astype(int).apply(
        lambda n: "Low" if n < 10 else ("Medium" if n < 30 else "High")
    )
    disp["rate"] = disp["rate"].round(1).astype(str) + "%"
    disp.columns = [level, marker, "Isolates tested", "Confidence"]
    st.dataframe(disp, hide_index=True, use_container_width=True)

# ---- Tab 7: Raw data ----
with tab7, tab_guard():
    view = st.radio("Table",
                     ["Isolates (one row per isolate)",
                      "AST results (long format — one row per antibiotic test)"],
                     horizontal=True)

    if view.startswith("Isolates"):
        st.markdown(f"**{len(iso_f):,}** isolates")
        cols = st.multiselect(
            "Columns",
            options=list(iso_f.columns),
            default=["isolate_id", "patient_id", "patient_age", "patient_sex",
                      "collection_date", "facility", "facility_type", "state",
                      "ward_type", "organism", "specimen_type",
                      "MRSA_status", "ESBL_status", "carbapenem_resistant"],
        )
        st.dataframe(iso_f[cols].head(500), use_container_width=True, hide_index=True)
        if len(iso_f) > 500:
            st.caption(f"Showing first 500 of {len(iso_f):,} rows.")
    else:
        st.markdown(f"**{len(ast_f):,}** AST results")
        cols = st.multiselect(
            "Columns",
            options=list(ast_f.columns),
            default=["isolate_id", "organism", "antibiotic", "antibiotic_class",
                      "interpretation", "facility", "facility_type",
                      "ward_type", "specimen_type", "collection_date"],
        )
        st.dataframe(ast_f[cols].head(500), use_container_width=True, hide_index=True)
        if len(ast_f) > 500:
            st.caption(f"Showing first 500 of {len(ast_f):,} rows.")

    st.markdown("---")
    st.info(
        "**Data access**  \n"
        "Data access for research, policy, or institutional use is available on request. "
        "Contact: hello@abimbolaoba.com"
    )

st.markdown("---")
st.caption(
    "Prototype — synthetic data calibrated to NCDC AMR surveillance and Nigerian literature "
    "(MRSA ~80%, ESBL 60–80%, carbapenem-R 20–30%). Public tertiary facilities only. "
    "Do not use for clinical or policy decisions."
)







