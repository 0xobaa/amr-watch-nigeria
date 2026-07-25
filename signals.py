"""
HealthScope Surveillance: signal queue.

Turns the landing screen into "what changed and does it need a decision"
instead of a set of static prevalence figures. Everything here is computed
from the live filtered dataframes, so it responds to the same filters as
every tab.

Depends only on streamlit, pandas, numpy, plotly. All already in requirements.

Severity tiers below are a working proposal, not a standard. Map them to
NCDC NAP 2.0 priority pathogen categories and have a microbiologist sign off
before this informs any real decision.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Agents with nothing meaningful behind them. Resistance here is a different
# kind of event from resistance to a first-line drug, which is why severity
# keys off drug class rather than off the percentage alone.
LAST_LINE_CLASSES = {"Polymyxins", "Carbapenems", "Glycopeptides", "Oxazolidinones"}

# Ordering within the Critical tier. Colistin failure outranks carbapenem
# failure even though the percentage is far smaller.
CLASS_PRIORITY = {"Polymyxins": 0, "Carbapenems": 1, "Glycopeptides": 2, "Oxazolidinones": 3}

SEVERITY = {
    "Critical":       {"rank": 0, "fg": "#E24B4A", "bg": "rgba(226,75,74,0.14)"},
    "High":           {"rank": 1, "fg": "#BA7517", "bg": "rgba(186,117,23,0.14)"},
    "Low confidence": {"rank": 2, "fg": "#8A8A85", "bg": "rgba(138,138,133,0.14)"},
}

# Class-specific critical thresholds. Treating every last-line detection as
# critical floods the queue: 2% vancomycin resistance is notable, 60% carbapenem
# resistance is an emergency, and a queue that calls both "Critical" is useless.
CRITICAL_AT = {
    "Polymyxins": 1.0,      # colistin: any confirmed resistance is critical
    "Carbapenems": 20.0,
    "Glycopeptides": 5.0,
    "Oxazolidinones": 5.0,
}

MIN_N_TO_SHOW = 30      # below this, not worth surfacing at all
MIN_N_CONFIDENT = 100   # below this, flag as low confidence
MAX_CI_HALFWIDTH = 7.5  # wider than this, the estimate cannot drive a decision


def _wilson(n_r: int, n: int, z: float = 1.96):
    """Wilson score interval. Returns (low, high) as percentages."""
    if n == 0:
        return 0.0, 0.0
    p = n_r / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom) * 100, min(1.0, (centre + margin) / denom) * 100


def _trend_delta(sub: pd.DataFrame):
    """Change in percentage points, first half of the period vs second half.

    Returns None when there is not enough history. Deliberately crude: it is a
    label on a card, not a forecast. The Resistance trends tab does the real work.
    """
    if "quarter" not in sub.columns:
        return None
    q = (sub.groupby("quarter")["interpretation"]
            .agg(n="count", r=lambda x: (x == "R").sum()))
    q = q[q["n"] >= 10]
    if len(q) < 6:
        return None
    rate = (q["r"] / q["n"] * 100).sort_index()
    half = len(rate) // 2
    return float(rate.iloc[half:].mean() - rate.iloc[:half].mean())


def detect_signals(ast_f: pd.DataFrame, abbrev: dict | None = None) -> list[dict]:
    """Score every organism-antibiotic pair in the current selection."""
    if ast_f is None or len(ast_f) == 0:
        return []
    abbrev = abbrev or {}

    grouped = (ast_f.groupby(["organism", "antibiotic", "antibiotic_class"])["interpretation"]
                    .agg(n="count", n_r=lambda x: (x == "R").sum())
                    .reset_index())
    grouped = grouped[grouped["n"] >= MIN_N_TO_SHOW]

    out = []
    for row in grouped.itertuples(index=False):
        n, n_r = int(row.n), int(row.n_r)
        rate = n_r / n * 100
        lo, hi = _wilson(n_r, n)
        halfwidth = (hi - lo) / 2
        last_line = row.antibiotic_class in LAST_LINE_CLASSES
        shaky = n < MIN_N_CONFIDENT or halfwidth > MAX_CI_HALFWIDTH

        detected = last_line and lo > 1.0   # interval clears zero, so it is real
        critical = detected and rate >= CRITICAL_AT.get(row.antibiotic_class, 100.0)

        if shaky:
            # Only report an uncertain cell if it would otherwise have been a
            # signal. Uncertain and unremarkable is just noise.
            if not detected and rate < 50:
                continue
            severity = "Low confidence"
        elif critical:
            severity = "Critical"
        elif detected or rate >= 50:
            severity = "High"
        else:
            continue

        org_short = abbrev.get(row.organism, row.organism)
        sub = ast_f[(ast_f["organism"] == row.organism) &
                    (ast_f["antibiotic"] == row.antibiotic)]
        delta = _trend_delta(sub)
        n_fac = sub["facility"].nunique() if "facility" in sub.columns else 0

        # Verdict is written from the data, not chosen from a template bank.
        if severity == "Low confidence":
            verdict = "Possible signal. Denominator too small to act on."
        elif row.antibiotic_class == "Polymyxins":
            verdict = "Last-line agent showing resistance. Nothing sits behind it."
        elif row.antibiotic_class == "Carbapenems":
            verdict = f"Carbapenem resistance at {rate:.1f}% in the current selection."
        elif row.antibiotic_class in ("Glycopeptides", "Oxazolidinones"):
            verdict = f"Reserve agent compromised in {org_short}."
        else:
            verdict = f"Empiric {row.antibiotic} is no longer defensible for {org_short}."

        bits = [f"{n_r:,} resistant of {n:,} tested"]
        if n_fac > 1:
            bits.append(f"across {n_fac} facilities")
        if delta is not None and abs(delta) >= 1.0:
            bits.append(f"{'up' if delta > 0 else 'down'} {abs(delta):.1f} points over the period")
        detail = ", ".join(bits) + "."
        if severity == "Low confidence":
            detail += f" Interval spans {hi - lo:.1f} points."

        out.append(dict(
            severity=severity, organism=row.organism, organism_short=org_short,
            antibiotic=row.antibiotic, antibiotic_class=row.antibiotic_class,
            rate=rate, lo=round(lo, 1), hi=round(hi, 1), n=n,
            verdict=verdict, detail=detail, delta=delta,
        ))

    out.sort(key=lambda s: (
        SEVERITY[s["severity"]]["rank"],
        CLASS_PRIORITY.get(s["antibiotic_class"], 9),
        -s["rate"],
    ))

    # Collapse to one row per organism and drug class. Meropenem, imipenem and
    # ertapenem failing in the same organism is one clinical event, not three.
    # The worst agent represents the class; the antibiogram tab has the rest.
    seen, deduped = set(), []
    for sig in out:
        key = (sig["organism"], sig["antibiotic_class"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sig)
    return deduped


def facility_outliers(iso_f: pd.DataFrame, enterobacterales) -> tuple[list[dict], float]:
    """Carbapenem-R among Enterobacterales, per facility, with Wilson intervals."""
    if iso_f is None or len(iso_f) == 0 or "carbapenem_resistant" not in iso_f.columns:
        return [], float("nan")
    ent = iso_f[iso_f["organism"].isin(enterobacterales)
                & iso_f["carbapenem_resistant"].isin(["Yes", "No"])]
    if len(ent) == 0:
        return [], float("nan")

    baseline = (ent["carbapenem_resistant"] == "Yes").mean() * 100
    rows = []
    for fac, g in ent.groupby("facility"):
        n = len(g)
        if n < MIN_N_TO_SHOW:
            continue
        n_r = int((g["carbapenem_resistant"] == "Yes").sum())
        lo, hi = _wilson(n_r, n)
        rows.append(dict(name=fac, short=fac.split(" (")[0], rate=n_r / n * 100,
                         lo=lo, hi=hi, n=n))
    rows.sort(key=lambda r: r["rate"], reverse=True)
    return rows, baseline


def _facility_figure(rows, baseline):
    def colour(r):
        if r["lo"] > baseline:
            return "#E24B4A"
        if r["hi"] < baseline:
            return "#639922"
        return "#8A8A85"

    fig = go.Figure(go.Scatter(
        x=[r["rate"] for r in rows],
        y=[r["short"] for r in rows],
        mode="markers",
        marker=dict(size=9, color=[colour(r) for r in rows]),
        error_x=dict(type="data", symmetric=False,
                     array=[r["hi"] - r["rate"] for r in rows],
                     arrayminus=[r["rate"] - r["lo"] for r in rows],
                     thickness=1.4, width=0),
        customdata=[[r["name"], r["n"], r["lo"], r["hi"]] for r in rows],
        hovertemplate=("%{customdata[0]}<br>%{x:.1f}% R"
                       "<br>95% CI %{customdata[2]:.1f}-%{customdata[3]:.1f}%"
                       "<br>n=%{customdata[1]:,}<extra></extra>"),
    ))
    fig.add_vline(x=baseline, line_width=1, line_dash="dot")
    # Height scales with row count so labels never collide on a narrow screen.
    fig.update_layout(
        height=max(220, 26 * len(rows) + 70),
        margin=dict(l=0, r=6, t=6, b=34),
        showlegend=False,
        xaxis_title=f"carbapenem-R Enterobacterales (%) · dotted line = {baseline:.1f}% network",
        font=dict(size=11),
    )
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    return fig


def inject_mobile_css():
    """Small-screen rules. Streamlit squeezes columns rather than stacking them
    below a certain width, which turns five KPI cards into five unreadable
    slivers on a phone. These rules wrap them two-per-row instead, tighten the
    page gutters, and stop long headings from overflowing."""
    st.markdown("""
    <style>
    .hs-card{border:1px solid rgba(130,130,130,0.28);padding:11px 13px;margin-bottom:9px;}
    .hs-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
    .hs-badge{font-size:11px;padding:2px 8px;border-radius:6px;white-space:nowrap;}
    .hs-org{font-size:14px;font-weight:600;}
    .hs-ab{font-size:12px;opacity:0.7;}
    .hs-rate{margin-left:auto;font-size:16px;font-weight:600;}
    .hs-verdict{font-size:13.5px;margin-top:6px;line-height:1.45;}
    .hs-detail{font-size:12px;opacity:0.72;margin-top:3px;line-height:1.4;}
    .hs-meta{font-size:11px;opacity:0.55;margin-top:6px;}
    @media (max-width: 640px){
      .block-container{padding:0.75rem 0.75rem 3rem !important;}
      h1{font-size:1.4rem !important;line-height:1.25 !important;}
      h2{font-size:1.1rem !important;}
      h3{font-size:1rem !important;}
      [data-testid="stHorizontalBlock"]{flex-wrap:wrap !important;gap:0.5rem !important;}
      [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]{
        min-width:calc(50% - 0.5rem) !important;flex:1 1 calc(50% - 0.5rem) !important;}
      [data-testid="stMetricValue"]{font-size:1.15rem !important;}
      [data-testid="stMetricLabel"]{font-size:0.72rem !important;}
      .stTabs [data-baseweb="tab"]{padding:0 9px !important;font-size:0.82rem !important;}
      .hs-rate{margin-left:0;}
      .hs-org{font-size:13.5px;}
    }
    </style>
    """, unsafe_allow_html=True)


def _card(sig):
    s = SEVERITY[sig["severity"]]
    st.markdown(
        f"""<div class="hs-card" style="border-left:3px solid {s['fg']};">
  <div class="hs-row">
    <span class="hs-badge" style="background:{s['bg']};color:{s['fg']};">{sig['severity']}</span>
    <span class="hs-org">{sig['organism_short']}</span>
    <span class="hs-ab">{sig['antibiotic']}</span>
    <span class="hs-rate" style="color:{s['fg']};">{sig['rate']:.1f}%</span>
  </div>
  <div class="hs-verdict">{sig['verdict']}</div>
  <div class="hs-detail">{sig['detail']}</div>
  <div class="hs-meta">n={sig['n']:,} &middot; 95% CI {sig['lo']}-{sig['hi']}%</div>
</div>""",
        unsafe_allow_html=True,
    )


def render(ast_f, iso_f, enterobacterales, abbrev, max_cards=6):
    """Draw the signal queue. Returns the signal list so callers can reuse it."""
    signals = detect_signals(ast_f, abbrev)

    st.subheader("Signals needing review")
    if not signals:
        st.success("No organism-antibiotic pair in this selection crosses the review "
                   "threshold. Widen the filters or check the tabs below for detail.")
    else:
        crit = [s for s in signals if s["severity"] == "Critical"]
        if crit:
            last_line = sorted({s["antibiotic_class"] for s in crit})
            st.error(f"{len(crit)} critical signal(s). Affected reserve classes: "
                     f"{', '.join(last_line)}.")
        st.caption("Ranked by consequence, not by size. "
                   "Opening a signal pre-selects it in the tabs below.")

        for i, sig in enumerate(signals[:max_cards]):
            _card(sig)
            if st.button(f"Open {sig['organism_short']} · {sig['antibiotic']}",
                         key=f"hs_sig_{i}", width="stretch"):
                st.session_state["hs_focus_organism"] = sig["organism"]
                st.session_state["hs_focus_antibiotic"] = sig["antibiotic"]
                st.rerun()

        if len(signals) > max_cards:
            with st.expander(f"Show {len(signals) - max_cards} more signal(s)"):
                for sig in signals[max_cards:]:
                    _card(sig)

    rows, baseline = facility_outliers(iso_f, enterobacterales)
    if rows and not np.isnan(baseline):
        st.subheader("Facility outliers")
        st.caption("Grey means the interval overlaps the network baseline. "
                   "Rank order between grey facilities is not meaningful.")
        st.plotly_chart(_facility_figure(rows, baseline), width="stretch",
                        config={"displayModeBar": False, "responsive": True})

    return signals
