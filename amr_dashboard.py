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
    "Escherichia coli":          0.25,
    "Klebsiella pneumoniae":     0.20,
    "Staphylococcus aureus":     0.18,
    "Pseudomonas aeruginosa":    0.10,
    "Acinetobacter baumannii":   0.07,
    "Enterococcus faecalis":     0.06,
    "Salmonella Typhi":          0.05,
    "Streptococcus pneumoniae":  0.04,
    "Proteus mirabilis":         0.03,
    "Enterobacter cloacae":      0.02,
}

SPECIMENS = {
    "Urine": 0.32, "Blood": 0.22, "Wound swab": 0.18, "Sputum": 0.10,
    "Pus": 0.08, "CSF": 0.04, "Stool": 0.03, "High vaginal": 0.03,
}

WARDS = {
    "Medical ward": 0.22, "Surgical ward": 0.18, "ICU": 0.14, "Paediatrics": 0.13,
    "A&E": 0.12, "Obstetrics": 0.09, "Outpatient": 0.08, "Burns unit": 0.04,
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

    return P


@st.cache_data
def generate_amr_dataset(n_isolates: int = 4500, seed: int = 42):
    rng = np.random.default_rng(seed)

    # Patient pool — ~18% of patients will have more than one isolate
    n_patients = int(n_isolates * 0.82)
    patient_ids = [f"PT-NG-{i+1:06d}" for i in range(n_patients)]
    patients = {
        pid: {
            "age": int(np.clip(rng.normal(38, 22), 0, 95)),
            "sex": str(rng.choice(["Male", "Female"], p=[0.48, 0.52])),
        }
        for pid in patient_ids
    }
    # Sampling weights determine how likely each patient is to be picked for a new isolate.
    # Most patients have weight 1 (single isolate); a minority get higher weights so they
    # appear multiple times in the output — simulating repeat cultures from the same patient,
    # which is common in hospital surveillance. The [1,1,1,2,3]/[0.7,0.1,0.05,0.1,0.05]
    # mix yields roughly 45-50% of patients contributing >1 isolate after normalisation.
    pw = rng.choice([1, 1, 1, 2, 3], size=n_patients, p=[0.7, 0.1, 0.05, 0.1, 0.05]).astype(float)
    pw /= pw.sum()

    fac_names = list(FACILITIES.keys())
    fw = np.array([FACILITIES[f]["weight"] for f in fac_names])
    fw /= fw.sum()

    start = datetime(2023, 1, 1)

    isolates, ast_rows = [], []
    for i in range(n_isolates):
        pid = str(rng.choice(patient_ids, p=pw))
        p = patients[pid]
        fac = str(rng.choice(fac_names, p=fw))
        fmeta = FACILITIES[fac]
        org = str(rng.choice(list(ORGANISMS.keys()), p=list(ORGANISMS.values())))
        spec = str(rng.choice(list(SPECIMENS.keys()), p=list(SPECIMENS.values())))
        ward = str(rng.choice(list(WARDS.keys()), p=list(WARDS.values())))

        cdate = start + timedelta(days=int(rng.integers(0, 730)))
        # Result turnaround varies by specimen type — blood cultures can take up to 7 days
        # for full AST panel, whereas urines may clear in 2-3 days. Sample 2-8 inclusive.
        rdate = cdate + timedelta(days=int(rng.integers(2, 9)))

        context_mult = (WARD_MULT[ward]
                         * FACILITY_TYPE_MULT[fmeta["type"]]
                         * (1 + 0.03 * (cdate.year - 2023)))

        iid = f"NG-{cdate.year}-{i+1:05d}"
        row = {
            "isolate_id": iid,
            "patient_id": pid,
            "patient_age": p["age"],
            "patient_age_group": _age_group(p["age"]),
            "patient_sex": p["sex"],
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
              "ward_type", "specimen_type", "collection_date", "year",
              "quarter", "patient_age_group", "patient_sex"]],
        on="isolate_id", how="left"
    )


ast_full = ast_with_meta(ast_df, isolates_df)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🧫 AMR Dashboard")
st.sidebar.caption("Nigerian Antimicrobial Resistance Surveillance — Prototype")

st.sidebar.subheader("Filters")
year_f = st.sidebar.multiselect("Year", sorted(isolates_df["year"].unique()),
                                  default=sorted(isolates_df["year"].unique()))
ftype_f = st.sidebar.multiselect("Facility type", sorted(isolates_df["facility_type"].unique()),
                                  default=sorted(isolates_df["facility_type"].unique()))
zone_f = st.sidebar.multiselect("Geopolitical zone", sorted(isolates_df["zone"].unique()),
                                  default=sorted(isolates_df["zone"].unique()))
fac_f = st.sidebar.multiselect("Facility", sorted(isolates_df["facility"].unique()),
                                default=sorted(isolates_df["facility"].unique()))
spec_f = st.sidebar.multiselect("Specimen type", sorted(isolates_df["specimen_type"].unique()),
                                 default=sorted(isolates_df["specimen_type"].unique()))
ward_f = st.sidebar.multiselect("Ward type", sorted(isolates_df["ward_type"].unique()),
                                 default=sorted(isolates_df["ward_type"].unique()))

iso_f = isolates_df[
    isolates_df["year"].isin(year_f)
    & isolates_df["facility_type"].isin(ftype_f)
    & isolates_df["zone"].isin(zone_f)
    & isolates_df["facility"].isin(fac_f)
    & isolates_df["specimen_type"].isin(spec_f)
    & isolates_df["ward_type"].isin(ward_f)
]
ast_f = ast_full[ast_full["isolate_id"].isin(iso_f["isolate_id"])]

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
# KPI row
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)

sa = iso_f[iso_f["organism"] == "Staphylococcus aureus"]
mrsa_rate = (sa["MRSA_status"] == "MRSA").mean() * 100 if len(sa) else np.nan

# ESBL denominator: only Enterobacterales that have an ESBL result. Requiring organism
# membership explicitly (rather than relying on ESBL_status != 'N/A') keeps this robust if
# we later add Enterobacterales species that don't have ESBL testing configured.
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

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["📊 Overview", "🧬 Organisms", "💊 Antibiograms", "📈 Resistance trends",
     "👥 Demographics", "🗺️ Geography", "📋 Raw data"]
)

# ---- Tab 1 ----
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Isolates over time")
        m = iso_f.groupby("month").size().reset_index(name="count").sort_values("month")
        fig = px.line(m, x="month", y="count", markers=True)
        fig.update_layout(xaxis_title="Month", yaxis_title="Isolates",
                           hovermode="x unified", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Specimen distribution")
        sp = iso_f["specimen_type"].value_counts().reset_index()
        sp.columns = ["specimen_type", "count"]
        sp["pct"] = (sp["count"] / sp["count"].sum() * 100).round(1)
        fig = px.bar(sp, x="count", y="specimen_type", orientation="h", text="pct",
                      hover_data=["count"])
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(xaxis_title="Isolates", yaxis_title="", height=380,
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Isolates by facility type")
        ft = iso_f["facility_type"].value_counts().reset_index()
        ft.columns = ["facility_type", "count"]
        fig = px.bar(ft, x="count", y="facility_type", orientation="h")
        fig.update_layout(xaxis_title="Isolates", yaxis_title="", height=380,
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.subheader("Isolates by ward")
        wd = iso_f["ward_type"].value_counts().reset_index()
        wd.columns = ["ward_type", "count"]
        fig = px.bar(wd, x="count", y="ward_type", orientation="h")
        fig.update_layout(xaxis_title="Isolates", yaxis_title="", height=380,
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Patient age by sex")
        fig = px.histogram(iso_f, x="patient_age", nbins=20, color="patient_sex",
                            barmode="overlay", opacity=0.7)
        fig.update_layout(xaxis_title="Age (years)", yaxis_title="Isolates", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        st.subheader("Resistance markers over time")
        t = iso_f.copy()
        t["period"] = t["collection_date"].dt.to_period("Q").astype(str)

        sa_t = t[t["organism"] == "Staphylococcus aureus"].groupby("period").apply(
            lambda g: (g["MRSA_status"] == "MRSA").mean() * 100, include_groups=False
        ).reset_index(name="MRSA %")
        esbl_t = t[t["organism"].isin(ENTEROBACTERALES)
                    & t["ESBL_status"].isin(["Positive", "Negative"])].groupby("period").apply(
            lambda g: (g["ESBL_status"] == "Positive").mean() * 100, include_groups=False
        ).reset_index(name="ESBL %")
        carb_t = t[t["carbapenem_resistant"].isin(["Yes", "No"])
                    & t["organism"].isin(ENTEROBACTERALES)].groupby("period").apply(
            lambda g: (g["carbapenem_resistant"] == "Yes").mean() * 100, include_groups=False
        ).reset_index(name="Carbapenem-R %")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sa_t["period"], y=sa_t["MRSA %"], mode="lines+markers", name="MRSA"))
        fig.add_trace(go.Scatter(x=esbl_t["period"], y=esbl_t["ESBL %"], mode="lines+markers", name="ESBL"))
        fig.add_trace(go.Scatter(x=carb_t["period"], y=carb_t["Carbapenem-R %"],
                                   mode="lines+markers", name="Carbapenem-R"))
        fig.update_layout(xaxis_title="Quarter", yaxis_title="% resistant",
                           hovermode="x unified", height=380, yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

# ---- Tab 2 ----
with tab2:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Pathogen frequency")
        oc = iso_f["organism"].value_counts().reset_index()
        oc.columns = ["organism", "count"]
        oc["pct"] = (oc["count"] / oc["count"].sum() * 100).round(1)
        fig = px.bar(oc, x="count", y="organism", orientation="h", text="pct")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(xaxis_title="Isolates", yaxis_title="", height=500,
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Counts")
        st.dataframe(oc, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Organism × specimen heatmap")
    piv = pd.crosstab(iso_f["organism"], iso_f["specimen_type"])
    fig = px.imshow(piv, aspect="auto", color_continuous_scale="Blues",
                     labels={"color": "Count"})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 3: Antibiograms — the big win of long format ----
with tab3:
    st.subheader("Antibiogram — resistance profile by organism")

    orgs = sorted(ast_f["organism"].unique())
    if not orgs:
        st.info("No AST data for current filter.")
    else:
        default_i = orgs.index("Escherichia coli") if "Escherichia coli" in orgs else 0
        sel = st.selectbox("Organism", orgs, index=default_i)

        oa = ast_f[ast_f["organism"] == sel]

        # Long format makes this a one-liner
        summary = (oa.groupby(["antibiotic", "antibiotic_class"])
                    .agg(n_tested=("interpretation", "count"),
                         n_r=("interpretation", lambda x: (x == "R").sum()))
                    .reset_index())
        summary["pct_r"] = (summary["n_r"] / summary["n_tested"] * 100).round(1)
        summary = summary[summary["n_tested"] >= 10].sort_values("pct_r")

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(summary, x="pct_r", y="antibiotic", orientation="h",
                          text="pct_r", color="pct_r",
                          color_continuous_scale="RdYlGn_r", range_color=[0, 100],
                          hover_data=["antibiotic_class", "n_tested"])
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(xaxis_title="% Resistant", yaxis_title="",
                               height=max(400, 40 * len(summary)),
                               xaxis_range=[0, 110], coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown(f"**{sel}** — {oa['isolate_id'].nunique():,} isolates")
            disp = summary[["antibiotic", "antibiotic_class", "pct_r", "n_tested"]].copy()
            disp.columns = ["Antibiotic", "Class", "%R", "n"]
            st.dataframe(disp, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Resistance by antibiotic class (across all organisms in selection)")
    cs = (ast_f.groupby("antibiotic_class")
            .agg(n_tested=("interpretation", "count"),
                 n_r=("interpretation", lambda x: (x == "R").sum()))
            .reset_index())
    cs["pct_r"] = (cs["n_r"] / cs["n_tested"] * 100).round(1)
    cs = cs.sort_values("pct_r")
    fig = px.bar(cs, x="pct_r", y="antibiotic_class", orientation="h", text="pct_r",
                  color="pct_r", color_continuous_scale="RdYlGn_r", range_color=[0, 100])
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(xaxis_title="% Resistant", yaxis_title="", height=400,
                       xaxis_range=[0, 110], coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 4: Resistance trends — the early-warning clinical view ----
with tab4:
    st.subheader("Resistance trend — organism × antibiotic over time")
    st.caption("Track how resistance to a specific antibiotic has evolved for a given organism. "
                "Useful for detecting emerging resistance and informing empirical therapy.")

    trend_orgs = sorted(ast_f["organism"].unique())
    if not trend_orgs:
        st.info("No AST data for current filter.")
    else:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            t_org = st.selectbox("Organism", trend_orgs,
                                  index=trend_orgs.index("Klebsiella pneumoniae")
                                  if "Klebsiella pneumoniae" in trend_orgs else 0,
                                  key="trend_org")

        # Antibiotics dynamically tied to the selected organism
        org_abs = sorted(ast_f[ast_f["organism"] == t_org]["antibiotic"].unique())
        with col_b:
            t_ab = st.selectbox("Antibiotic", org_abs,
                                 index=org_abs.index("Meropenem")
                                 if "Meropenem" in org_abs else 0,
                                 key="trend_ab")
        with col_c:
            period = st.radio("Period", ["Monthly", "Quarterly"], horizontal=True,
                              key="trend_period")

        # Optional facility comparison
        compare = st.checkbox("Compare across facilities", value=False)
        chosen_facs = None
        skip_chart = False
        if compare:
            # Default to the 4 highest-volume facilities in the current selection, so the
            # comparison starts with statistically meaningful signal rather than the 4 sites
            # that happen to come first alphabetically.
            vol_ranked = (ast_f[ast_f["organism"] == t_org]["facility"]
                           .value_counts().index.tolist())
            chosen_facs = st.multiselect("Facilities to compare", vol_ranked,
                                          default=vol_ranked[:4])
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
            trend_summary["pct_r"] = (trend_summary["n_r"] / trend_summary["n_tested"] * 100).round(1)

            # Sort period chronologically
            if period == "Monthly":
                trend_summary = trend_summary.sort_values("period")
            else:
                trend_summary["_sort"] = trend_summary["period"].apply(
                    lambda s: (int(s.split()[1]), int(s.split()[0][1:]))
                )
                trend_summary = trend_summary.sort_values("_sort").drop(columns="_sort")

            # Main trend chart
            if compare:
                fig = px.line(trend_summary, x="period", y="pct_r", color="facility",
                               markers=True, hover_data=["n_tested"])
            else:
                fig = px.line(trend_summary, x="period", y="pct_r", markers=True,
                               hover_data=["n_tested"])
            fig.update_layout(
                xaxis_title="Month" if period == "Monthly" else "Quarter",
                yaxis_title=f"% Resistant to {t_ab}",
                yaxis_range=[0, 100], hovermode="x unified", height=450,
                title=f"{t_org} — {t_ab} resistance trend",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary below
            col_x, col_y, col_z = st.columns(3)
            overall_r = (t_data["interpretation"] == "R").mean() * 100
            total_tested = len(t_data)

            # Latest-period %R — compute differently depending on compare mode
            period_label = "month" if period == "Monthly" else "quarter"
            if compare:
                # Pool all facilities' tests in the latest period for a true weighted rate,
                # rather than averaging facility-level percentages (which would over-weight
                # low-volume sites).
                latest_period = trend_summary["period"].iloc[-1]
                latest_rows = trend_summary[trend_summary["period"] == latest_period]
                latest_pct = (latest_rows["n_r"].sum() / latest_rows["n_tested"].sum() * 100) \
                             if latest_rows["n_tested"].sum() > 0 else float("nan")
                latest_label = f"Latest {period_label} %R (pooled)"
                # len(latest_rows) = facilities with data in the latest period, which may be
                # smaller than len(chosen_facs) if some sites had no tests that period.
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

            col_x.metric("Overall %R", f"{overall_r:.1f}%")
            col_y.metric("Total tests", f"{total_tested:,}")
            col_z.metric(latest_label, f"{latest_pct:.1f}%", help=latest_help)

            with st.expander("Data table"):
                show = trend_summary.copy()
                # Explicit column rename — cleaner than blanket title-casing, which produced
                # awkward labels like "N R" and "Pct R".
                rename_map = {
                    "period": "Period",
                    "facility": "Facility",
                    "n_tested": "n tested",
                    "n_r": "n resistant",
                    "pct_r": "%R",
                }
                show = show.rename(columns=rename_map)
                st.dataframe(show, hide_index=True, use_container_width=True)

# ---- Tab 5: Demographics ----
with tab5:
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
                                  index=demo_orgs.index("Escherichia coli")
                                  if "Escherichia coli" in demo_orgs else 0,
                                  key="demo_org")
        org_abs = sorted(ast_f[ast_f["organism"] == d_org]["antibiotic"].unique())
        with col_b:
            d_ab = st.selectbox("Antibiotic", org_abs,
                                 index=org_abs.index("Ciprofloxacin")
                                 if "Ciprofloxacin" in org_abs else 0,
                                 key="demo_ab")

        d_data = ast_f[(ast_f["organism"] == d_org) & (ast_f["antibiotic"] == d_ab)].copy()

        # Age group ordering used across all three charts below
        age_order = ["<1 year", "1-4", "5-14", "15-44", "45-64", "65+"]

        if len(d_data) == 0:
            st.warning("No records for this combination.")
        else:
            c1, c2 = st.columns(2)

            # By age group
            with c1:
                st.markdown("**Resistance by age group**")
                age_sum = (d_data.groupby("patient_age_group")
                            .agg(n=("interpretation", "count"),
                                 n_r=("interpretation", lambda x: (x == "R").sum()))
                            .reset_index())
                age_sum["pct_r"] = (age_sum["n_r"] / age_sum["n"] * 100).round(1)
                age_sum["patient_age_group"] = pd.Categorical(
                    age_sum["patient_age_group"], categories=age_order, ordered=True
                )
                age_sum = age_sum.sort_values("patient_age_group")

                fig = px.bar(age_sum, x="patient_age_group", y="pct_r", text="pct_r",
                              color="pct_r", color_continuous_scale="RdYlGn_r",
                              range_color=[0, 100], hover_data=["n"])
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
                fig.update_layout(xaxis_title="Age group",
                                   yaxis_title=f"% Resistant to {d_ab}",
                                   yaxis_range=[0, 110], height=400,
                                   coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            # By sex
            with c2:
                st.markdown("**Resistance by sex**")
                sex_sum = (d_data.groupby("patient_sex")
                            .agg(n=("interpretation", "count"),
                                 n_r=("interpretation", lambda x: (x == "R").sum()))
                            .reset_index())
                sex_sum["pct_r"] = (sex_sum["n_r"] / sex_sum["n"] * 100).round(1)

                fig = px.bar(sex_sum, x="patient_sex", y="pct_r", text="pct_r",
                              color="patient_sex", hover_data=["n"])
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
                fig.update_layout(xaxis_title="Sex",
                                   yaxis_title=f"% Resistant to {d_ab}",
                                   yaxis_range=[0, 110], height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Combined age × sex
            st.markdown("**Resistance by age group × sex**")
            combo = (d_data.groupby(["patient_age_group", "patient_sex"])
                      .agg(n=("interpretation", "count"),
                           n_r=("interpretation", lambda x: (x == "R").sum()))
                      .reset_index())
            combo["pct_r"] = (combo["n_r"] / combo["n"] * 100).round(1)
            combo["patient_age_group"] = pd.Categorical(
                combo["patient_age_group"], categories=age_order, ordered=True
            )
            combo = combo.sort_values("patient_age_group")

            fig = px.bar(combo, x="patient_age_group", y="pct_r", color="patient_sex",
                          barmode="group", text="pct_r", hover_data=["n"])
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(xaxis_title="Age group",
                               yaxis_title=f"% Resistant to {d_ab}",
                               yaxis_range=[0, 110], height=420)
            st.plotly_chart(fig, use_container_width=True)

            # Sample size warning
            small_cells = combo[combo["n"] < 10]
            if len(small_cells) > 0:
                st.warning(
                    f"⚠️ {len(small_cells)} age × sex cells have fewer than 10 tests. "
                    "Interpret those rates with caution."
                )

            with st.expander("Data table"):
                show = combo[["patient_age_group", "patient_sex", "n", "n_r", "pct_r"]].copy()
                show.columns = ["Age group", "Sex", "n tested", "n resistant", "%R"]
                st.dataframe(show, hide_index=True, use_container_width=True)

# ---- Tab 6: Geography ----
with tab6:
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
    disp["rate"] = disp["rate"].round(1).astype(str) + "%"
    disp.columns = [level, marker, "n"]
    st.dataframe(disp, hide_index=True, use_container_width=True)

# ---- Tab 7: Raw data ----
with tab7:
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
            st.caption(f"Showing first 500 of {len(iso_f):,} rows. "
                        f"Download the CSV to access all {len(iso_f):,} isolates.")
        st.download_button("⬇️ Download isolates CSV",
                            iso_f.to_csv(index=False).encode("utf-8"),
                            f"ng_amr_isolates_{datetime.now().strftime('%Y%m%d')}.csv",
                            "text/csv")
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
            st.caption(f"Showing first 500 of {len(ast_f):,} rows. "
                        f"Download the CSV to access all {len(ast_f):,} AST results.")
        st.download_button("⬇️ Download AST long-format CSV",
                            ast_f.to_csv(index=False).encode("utf-8"),
                            f"ng_amr_ast_long_{datetime.now().strftime('%Y%m%d')}.csv",
                            "text/csv")

st.markdown("---")
st.caption(
    "Prototype — synthetic data calibrated to NCDC AMR surveillance and Nigerian literature "
    "(MRSA ~80%, ESBL 60–80%, carbapenem-R 20–30%). Public tertiary facilities only. "
    "Do not use for clinical or policy decisions."
)
