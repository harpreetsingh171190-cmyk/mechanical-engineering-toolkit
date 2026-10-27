"""Unified Mechanical Engineering Dashboard.
Run with: streamlit run web_dashboard.py
Educational calculations only; validate assumptions and applicable design codes.
"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
import math
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import engineering_toolkit as et
from mohr_circle import MohrsCircle
from projectile_motion import projectile_trajectory
from mechanical_ds.analytics import (
    detect_anomalies, eda_summary, forecast_metrics,
    moving_average_forecast, optimize_beam_depth, simulate_mass_spring,
)
from mechanical_ds.storage import HistoryStore

st.set_page_config(page_title="Mechanical Engineering — All-in-One", page_icon="⚙️", layout="wide")
st.title("⚙️ Mechanical Engineering — All-in-One Dashboard")
st.caption("Calculators • interactive graphs • CSV lab • ML training • simulations • PDF reports")
st.warning("Educational tool only. Results are model-dependent and are not a substitute for engineering review, safety factors, standards, or qualified sign-off.")

@st.cache_resource
def history_store():
    return HistoryStore()

store = history_store()
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "latest_ml" not in st.session_state:
    st.session_state.latest_ml = None


def serializable(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, dict): return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serializable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    return str(value)


def record_activity(kind: str, result: dict):
    item = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "result": serializable(result)}
    st.session_state.activity_log.append(item)
    try:
        store.add(kind, item["result"])
    except Exception:
        pass


def make_pdf(title: str, sections: list[tuple[str, object]]) -> bytes:
    """Build a compact PDF report with ReportLab; no temporary files required."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

    buff = BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#555555")))
    story = [Paragraph(title, styles["Title"]), Spacer(1, 4*mm),
             Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["SmallNote"]), Spacer(1, 5*mm)]
    for heading, content in sections:
        story.append(Paragraph(str(heading), styles["Heading2"]))
        if isinstance(content, pd.DataFrame):
            frame = content.copy().head(30).fillna("")
            rows = [[str(c)[:35] for c in frame.columns]] + [[str(v)[:45] for v in row] for row in frame.astype(str).values.tolist()]
            if len(rows[0]) > 6: rows = [[str(c)[:22] for c in frame.columns[:6]]] + [[str(v)[:28] for v in row[:6]] for row in frame.astype(str).values.tolist()]
            table = Table(rows, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#263746")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 7), ("GRID", (0,0), (-1,-1), .25, colors.grey), ("VALIGN", (0,0), (-1,-1), "TOP"), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F2F5F7")])]))
            story.append(table)
        elif isinstance(content, dict):
            rows = [[str(k), str(v)[:180]] for k, v in content.items()]
            table = Table(rows, colWidths=[55*mm, 115*mm], hAlign="LEFT")
            table.setStyle(TableStyle([("GRID", (0,0), (-1,-1), .25, colors.grey), ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8), ("VALIGN", (0,0), (-1,-1), "TOP"), ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#F2F5F7")])]))
            story.append(table)
        else:
            story.append(Paragraph(str(content).replace("&", "&amp;"), styles["BodyText"]))
        story.append(Spacer(1, 4*mm))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("Limitations: confirm units, assumptions, input data quality, applicable standards and results independently before any real-world use.", styles["SmallNote"]))
    doc.build(story)
    return buff.getvalue()


def show_result(title: str, result: dict):
    st.subheader(title)
    st.json(serializable(result))
    record_activity(title, result)


tabs = st.tabs(["🏠 Overview", "🧮 Calculators", "📈 Live Graphs & CSV", "🤖 ML Training", "🌀 Simulations", "📄 PDF Reports & History"])

with tabs[0]:
    st.subheader("One place for the complete engineering workflow")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Engineering calculators", "45+")
    c2.metric("Data workflow", "CSV / EDA / anomalies")
    c3.metric("ML models", "3 regressors")
    c4.metric("Exports", "CSV + PDF + model")
    st.markdown("""
    - **Calculators:** thermal expansion, torque, spring frequency, factor of safety, beam reactions/SFD/BMD and deflection (single, multi-load, fixed-fixed, propped cantilever and continuous multi-span), fatigue (modified Goodman and finite-life S-N), stress concentration (Kt and Kf), heat exchangers (LMTD and effectiveness-NTU), pipe-flow pressure drop, rolling-bearing life, bolted joints, straight fins, internal-flow convection, radiation, damped vibration and isolation, spur-gear geometry, stress and module sizing, section properties (rectangular and circular), Euler buckling, von Mises stress, shaft torsion, pressure vessels, unit conversion, Reynolds number, pump power, heat conduction, gear train, heat-engine and Carnot efficiency, beam-depth sizing, Mohr's circle and projectile motion.
    - **Data lab:** import CSV, select columns, inspect missing values and statistics, create interactive plots, flag outliers and build baseline forecasts.
    - **Machine learning:** train/test split, selectable regression algorithms, R² / MAE / RMSE, actual-vs-predicted plot, prediction CSV and downloadable model.
    - **Reports:** create PDF summaries of the latest calculation, dataset analysis, ML model, simulations and session activity.
    """)
    st.info("Start from the Calculators tab for engineering calculations, or upload your own measurement CSV in Live Graphs & CSV.")

with tabs[1]:
    calc = st.selectbox("Choose calculator", [
        "Thermal expansion", "Shaft torque", "Spring natural frequency", "Factor of safety",
        "Simply supported beam (point load)", "Unit conversion", "Reynolds number", "Pump power",
        "Heat conduction", "Gear train", "Heat-engine performance", "Carnot efficiency", "Simply supported beam (uniform load)", "Cantilever beam (end load)", "Cantilever beam (uniform load)", "Multi-load beam", "Fixed–fixed beam", "Continuous beam", "Rectangular section properties", "Beam deflection (point load)", "Euler column buckling", "Von Mises stress", "Solid shaft torsion", "Thin-wall pressure vessel", "Fatigue (modified Goodman)", "Stress concentration (Kt, Kf)", "S-N fatigue life", "Heat exchanger (LMTD)", "Heat exchanger (effectiveness-NTU)", "Pipe flow pressure drop", "Rolling-bearing life (L10)", "Bolted joint", "Straight fin", "Internal-flow convection (Nusselt)", "Circular section properties", "Radiation heat transfer", "Damped vibration", "Spur gear geometry", "Spur gear stress and sizing", "Rectangular beam depth sizing", "Mohr’s circle / principal stress", "Projectile trajectory", "Compression spring design", "Belt drive design", "Power screw design", "Shaft design (fatigue)",
    ])
    NTU_LABELS = {"Counterflow": "counterflow", "Parallel flow": "parallel_flow", "Shell-and-tube (1 shell pass)": "shell_and_tube",
                  "Crossflow, both fluids unmixed": "crossflow_both_unmixed", "Crossflow, Cmax mixed": "crossflow_cmax_mixed",
                  "Crossflow, Cmin mixed": "crossflow_cmin_mixed"}
    UNIT_MAP = {"length": ["m", "mm", "cm", "km", "in", "ft"], "force": ["N", "kN", "lbf"],
                "pressure": ["Pa", "kPa", "MPa", "GPa", "bar", "psi"], "energy": ["J", "kJ", "cal", "kWh"],
                "power": ["W", "kW", "hp"], "temperature": ["C", "F", "K"]}
    if calc == "Unit conversion":  # outside the form so the From/To lists update immediately
        unit_group = st.selectbox("Unit family", list(UNIT_MAP))
    with st.form("calculator_form"):
        if calc == "Thermal expansion":
            a,b,c = st.columns(3); L=a.number_input("Initial length (m)", min_value=0.0, value=2.0); alpha=b.number_input("Expansion coefficient (1/°C)", min_value=0.0, value=0.000012, format="%.8f"); dt=c.number_input("Temperature change (°C)", value=60.0)
        elif calc == "Shaft torque":
            a,b=st.columns(2); power=a.number_input("Power (W)", min_value=0.0, value=5000.0); rpm=b.number_input("Speed (RPM)", min_value=0.1, value=1500.0)
        elif calc == "Spring natural frequency":
            a,b=st.columns(2); k=a.number_input("Stiffness (N/m)", min_value=0.001, value=1000.0); mass=b.number_input("Mass (kg)", min_value=0.001, value=10.0)
        elif calc == "Factor of safety":
            a,b=st.columns(2); strength=a.number_input("Failure strength (Pa)", min_value=0.001, value=250e6, format="%.3f"); stress=b.number_input("Working stress (Pa)", min_value=0.001, value=100e6, format="%.3f")
        elif calc == "Simply supported beam (point load)":
            a,b,c=st.columns(3); span=a.number_input("Span (m)", min_value=0.001, value=4.0); load=b.number_input("Point load (N)", min_value=0.0, value=1000.0); pos=c.number_input("Load position from left (m)", min_value=0.0, value=float(span/2), key="beam_position")
        elif calc == "Unit conversion":
            val=st.number_input("Value", value=1.0); a,b=st.columns(2); from_unit=a.selectbox("From",UNIT_MAP[unit_group]); to_unit=b.selectbox("To",UNIT_MAP[unit_group],index=min(1,len(UNIT_MAP[unit_group])-1))
        elif calc == "Reynolds number":
            a,b,c,d=st.columns(4); rho=a.number_input("Density (kg/m³)", min_value=.000001, value=1000.0); velocity=b.number_input("Velocity (m/s)", min_value=0.0, value=1.0); diameter=c.number_input("Characteristic diameter (m)", min_value=.000001, value=.05); mu=d.number_input("Dynamic viscosity (Pa·s)", min_value=.000001, value=.001, format="%.6f")
        elif calc == "Pump power":
            a,b,c=st.columns(3); flow=a.number_input("Flow rate (m³/s)", min_value=0.0, value=.02, format="%.5f"); dp=b.number_input("Pressure rise (Pa)", min_value=0.0, value=200000.0); eta=c.number_input("Efficiency (0–1)", min_value=.01, max_value=1.0, value=.75)
        elif calc == "Heat conduction":
            a,b,c=st.columns(3); conductivity=a.number_input("Conductivity k (W/m·K)", min_value=.000001, value=45.0); area=b.number_input("Area (m²)", min_value=.000001, value=2.0); thickness=c.number_input("Wall thickness (m)", min_value=.000001, value=.1); d,e=st.columns(2); hot=d.number_input("Hot-side temperature (°C)", value=150.0); cold=e.number_input("Cold-side temperature (°C)", value=30.0)
        elif calc == "Gear train":
            a,b,c,d,e=st.columns(5); rpm=a.number_input("Input RPM", min_value=0.0, value=1200.0); driver=b.number_input("Driver teeth", min_value=1, value=20, step=1); driven=c.number_input("Driven teeth", min_value=1, value=60, step=1); torque=d.number_input("Input torque (N·m)", min_value=0.0, value=25.0); eta=e.number_input("Efficiency (0–1)", min_value=.01, max_value=1.0, value=.95)
        elif calc == "Heat-engine performance":
            a,b=st.columns(2); qin=a.number_input("Heat input rate (W)", min_value=.001, value=100000.0); qout=b.number_input("Heat rejected rate (W)", min_value=0.0, value=60000.0)
        elif calc == "Carnot efficiency":
            a,b=st.columns(2); th=a.number_input("Hot reservoir (K)", min_value=.001, value=800.0); tc=b.number_input("Cold reservoir (K)", min_value=.001, value=300.0)
        elif calc in ("Simply supported beam (uniform load)", "Cantilever beam (end load)", "Cantilever beam (uniform load)"):
            point = calc == "Cantilever beam (end load)"
            a,b=st.columns(2); beam_len=a.number_input("Span / length (m)", min_value=.001, value=4.0, key="b_len"); beam_load=b.number_input("Point load (N)" if point else "Uniform load (N/m)", min_value=0.0, value=5000.0, key="b_load"); d,e=st.columns(2); beam_e=d.number_input("Young's modulus E (GPa)", min_value=.001, value=200.0, key="b_e"); beam_i=e.number_input("Second moment I (cm⁴)", min_value=.001, value=2000.0, key="b_i")
        elif calc == "Multi-load beam":
            a,b=st.columns(2); ml_support=a.selectbox("Support", ["Simply supported","Cantilever (fixed at left end)","Fixed–fixed","Propped cantilever (fixed left, pin right)"], key="ml_support"); ml_len=b.number_input("Span / length (m)", min_value=.001, value=6.0, key="ml_len")
            ml_points=st.text_area("Point loads: position (m), load (N), one per line (downward positive)", "1.5, 2000\n4.5, 3000", key="ml_points")
            ml_udl=st.text_area("Uniform loads: start (m), end (m), w (N/m), one per line", "0, 6, 500", key="ml_udl")
            d,e=st.columns(2); ml_e=d.number_input("Young's modulus E (GPa)", min_value=.001, value=200.0, key="ml_e"); ml_i=e.number_input("Second moment I (cm⁴)", min_value=.001, value=2000.0, key="ml_i")
        elif calc == "Continuous beam":
            a,b,c=st.columns(3); cb_spans=a.text_input("Span lengths (m), comma separated", "4, 5, 4", key="cb_spans"); cb_left=b.selectbox("Left end", ["Pinned","Fixed","Free (overhang)"], key="cb_left"); cb_right=c.selectbox("Right end", ["Pinned","Fixed","Free (overhang)"], key="cb_right")
            cb_points=st.text_area("Point loads: position (m from the left end), load (N), one per line", "2, 10000", key="cb_points")
            cb_udl=st.text_area("Uniform loads: start (m), end (m), w (N/m), one per line", "0, 13, 5000", key="cb_udl")
            d,e=st.columns(2); cb_e=d.number_input("Young's modulus E (GPa)", min_value=.001, value=200.0, key="cb_e"); cb_i=e.number_input("Second moment I (cm⁴), same for every span", min_value=.001, value=8000.0, key="cb_i")
        elif calc == "Fixed–fixed beam":
            a,b=st.columns(2); ff_len=a.number_input("Span (m)", min_value=.001, value=4.0, key="ff_len"); ff_kind=b.selectbox("Load type", ["Point load","Uniform load (full span)"], key="ff_kind")
            c,d=st.columns(2); ff_load=c.number_input("Load (N for a point load, N/m for a uniform load)", min_value=0.0, value=5000.0, key="ff_load"); ff_pos=d.number_input("Point-load position from left (m)", min_value=0.0, value=2.0, key="ff_pos")
            e,f=st.columns(2); ff_e=e.number_input("Young's modulus E (GPa)", min_value=.001, value=200.0, key="ff_e"); ff_i=f.number_input("Second moment I (cm⁴)", min_value=.001, value=2000.0, key="ff_i")
        elif calc == "Fatigue (modified Goodman)":
            a,b,c=st.columns(3); fg_sut=a.number_input("Ultimate strength Sut (MPa)", min_value=.001, value=600.0, key="fg_sut"); fg_se=b.number_input("Corrected endurance limit Se (MPa)", min_value=.001, value=200.0, key="fg_se"); fg_sy=c.number_input("Yield strength Sy (MPa, 0 to skip)", min_value=0.0, value=400.0, key="fg_sy")
            d,e=st.columns(2); fg_max=d.number_input("Maximum stress (MPa)", value=300.0, key="fg_max"); fg_min=e.number_input("Minimum stress (MPa)", value=-100.0, key="fg_min")
        elif calc == "Stress concentration (Kt, Kf)":
            a,b=st.columns(2); sc_geo=a.selectbox("Notch geometry", ["Circular hole, finite-width plate","Elliptical hole, wide plate","Enter Kt from a chart"], key="sc_geo"); sc_nom=b.number_input("Nominal stress (MPa, 0 to skip peak stress)", min_value=0.0, value=100.0, key="sc_nom")
            c,d,e=st.columns(3); sc_w=c.number_input("Plate width w (mm)", min_value=.001, value=100.0, key="sc_w"); sc_d=d.number_input("Hole diameter d (mm)", min_value=.001, value=20.0, key="sc_d"); sc_kt=e.number_input("Kt (used for 'Enter Kt')", min_value=1.0, value=2.5, key="sc_kt")
            f,g=st.columns(2); sc_a=f.number_input("Ellipse semi-axis across the load a (mm)", min_value=.001, value=10.0, key="sc_a"); sc_b=g.number_input("Ellipse semi-axis along the load b (mm)", min_value=.001, value=5.0, key="sc_b")
            h,i,j,k=st.columns(4); sc_r=h.number_input("Notch radius (mm, 0 to skip Kf; d/2 for a hole)", min_value=0.0, value=10.0, key="sc_r"); sc_sut=i.number_input("Ultimate strength Sut (MPa)", min_value=.001, value=800.0, key="sc_sut"); sc_load=j.selectbox("Loading", ["Bending","Axial","Torsion"], key="sc_load"); sc_method=k.selectbox("Notch-sensitivity model", ["Neuber","Peterson"], key="sc_method")
        elif calc == "S-N fatigue life":
            a,b,c=st.columns(3); sn_sut=a.number_input("Ultimate strength Sut (MPa)", min_value=.001, value=600.0, key="sn_sut"); sn_se=b.number_input("Corrected endurance limit Se (MPa)", min_value=.001, value=200.0, key="sn_se"); sn_kf=c.number_input("Fatigue notch factor Kf (1 for none)", min_value=1.0, value=1.0, key="sn_kf")
            d,e=st.columns(2); sn_max=d.number_input("Maximum nominal stress (MPa)", value=300.0, key="sn_max"); sn_min=e.number_input("Minimum nominal stress (MPa)", value=-100.0, key="sn_min")
            f,g=st.columns(2); sn_f=f.number_input("Strength fraction f at 10³ cycles", min_value=.01, max_value=1.0, value=0.9, key="sn_f"); sn_target=g.number_input("Target life in cycles (0 to skip, otherwise ≥ 1000)", min_value=0.0, value=100000.0, key="sn_target")
        elif calc == "Heat exchanger (LMTD)":
            a,b,c,d=st.columns(4); hx_thi=a.number_input("Hot inlet (°C)", value=100.0, key="hx_thi"); hx_tho=b.number_input("Hot outlet (°C)", value=60.0, key="hx_tho"); hx_tci=c.number_input("Cold inlet (°C)", value=20.0, key="hx_tci"); hx_tco=d.number_input("Cold outlet (°C)", value=50.0, key="hx_tco")
            e,f=st.columns(2); hx_arr=e.selectbox("Flow arrangement", ["Counterflow","Parallel flow","Shell-and-tube (1 shell pass, 2+ tube passes)"], key="hx_arr"); hx_mode=f.selectbox("Solve for", ["LMTD only","Heat duty (from U and area)","Required area (from U and duty)"], key="hx_mode")
            g,h,i=st.columns(3); hx_u=g.number_input("Overall U (W/m²·K)", min_value=.001, value=500.0, key="hx_u"); hx_area=h.number_input("Area (m²)", min_value=.001, value=3.0, key="hx_area"); hx_q=i.number_input("Heat duty (W)", min_value=.001, value=50000.0, key="hx_q")
        elif calc == "Heat exchanger (effectiveness-NTU)":
            a,b,c,d=st.columns(4); nt_ch=a.number_input("Hot capacity rate m·cp (W/K)", min_value=.001, value=2000.0, key="nt_ch"); nt_cc=b.number_input("Cold capacity rate m·cp (W/K)", min_value=.001, value=4000.0, key="nt_cc"); nt_thi=c.number_input("Hot inlet (°C)", value=100.0, key="nt_thi"); nt_tci=d.number_input("Cold inlet (°C)", value=20.0, key="nt_tci")
            e,f=st.columns(2); nt_arr=e.selectbox("Flow arrangement", list(NTU_LABELS), key="nt_arr"); nt_mode=f.selectbox("Problem", ["Rating: given UA","Sizing: given effectiveness"], key="nt_mode")
            g,h=st.columns(2); nt_ua=g.number_input("UA (W/K)", min_value=.001, value=3000.0, key="nt_ua"); nt_eps=h.number_input("Target effectiveness (0–0.999)", min_value=0.0, max_value=0.999, value=0.7, key="nt_eps")
        elif calc == "Pipe flow pressure drop":
            a,b,c,d=st.columns(4); pp_d=a.number_input("Inside diameter (mm)", min_value=.001, value=50.0, key="pp_d"); pp_len=b.number_input("Pipe length (m)", min_value=0.0, value=100.0, key="pp_len"); pp_rho=c.number_input("Density (kg/m³)", min_value=.001, value=998.0, key="pp_rho"); pp_mu=d.number_input("Dynamic viscosity (mPa·s)", min_value=.0001, value=1.0, format="%.4f", key="pp_mu")
            e,f,g,h=st.columns(4); pp_mode=e.selectbox("Flow given as", ["Flow rate (L/s)","Velocity (m/s)"], key="pp_mode"); pp_flow=f.number_input("Flow value", min_value=0.0, value=4.0, key="pp_flow"); pp_rough=g.number_input("Roughness (mm)", min_value=0.0, value=0.045, format="%.4f", key="pp_rough"); pp_k=h.number_input("Sum of minor-loss K", min_value=0.0, value=3.0, key="pp_k")
            pp_dz=st.number_input("Elevation gain, inlet to outlet (m)", value=0.0, key="pp_dz")
        elif calc == "Rolling-bearing life (L10)":
            a,b,c=st.columns(3); br_c=a.number_input("Dynamic load rating C (kN)", min_value=.001, value=30.0, key="br_c"); br_fr=b.number_input("Radial load Fr (kN)", min_value=0.0, value=4.0, key="br_fr"); br_fa=c.number_input("Axial load Fa (kN)", min_value=0.0, value=1.0, key="br_fa")
            d,e,f=st.columns(3); br_x=d.number_input("Radial factor X", min_value=0.0, value=0.56, key="br_x"); br_y=e.number_input("Axial factor Y", min_value=0.0, value=1.6, key="br_y"); br_type=f.selectbox("Bearing type", ["Ball","Roller"], key="br_type")
            g,h,i=st.columns(3); br_rpm=g.number_input("Speed (RPM)", min_value=.001, value=1500.0, key="br_rpm"); br_rel=h.number_input("Reliability (%)", min_value=90.0, max_value=99.0, value=90.0, key="br_rel"); br_target=i.number_input("Target life (hours, 0 to skip)", min_value=0.0, value=20000.0, key="br_target")
        elif calc == "Bolted joint":
            a,b,c,d=st.columns(4); bj_d=a.number_input("Nominal diameter (mm)", min_value=.001, value=10.0, key="bj_d"); bj_p=b.number_input("Thread pitch (mm)", min_value=.001, value=1.5, key="bj_p"); bj_sp=c.number_input("Proof strength Sp (MPa)", min_value=.001, value=580.0, key="bj_sp"); bj_c=d.number_input("Joint constant C", min_value=.01, max_value=.99, value=0.25, key="bj_c")
            e,f,g=st.columns(3); bj_load=e.number_input("External tensile load (kN)", min_value=0.0, value=8.0, key="bj_load"); bj_frac=f.number_input("Preload fraction of proof load", min_value=.05, max_value=1.0, value=0.75, key="bj_frac"); bj_k=g.number_input("Torque coefficient K", min_value=.01, value=0.2, key="bj_k")
        elif calc == "Straight fin":
            a,b,c,d=st.columns(4); fn_k=a.number_input("Conductivity k (W/m·K)", min_value=.001, value=200.0, key="fn_k"); fn_h=b.number_input("Convection h (W/m²·K)", min_value=.001, value=50.0, key="fn_h"); fn_len=c.number_input("Length (mm)", min_value=.001, value=50.0, key="fn_len"); fn_t=d.number_input("Thickness (mm)", min_value=.001, value=4.0, key="fn_t")
            e,f,g,h=st.columns(4); fn_w=e.number_input("Width (mm)", min_value=.001, value=100.0, key="fn_w"); fn_tb=f.number_input("Base temperature (°C)", value=100.0, key="fn_tb"); fn_ta=g.number_input("Ambient temperature (°C)", value=25.0, key="fn_ta"); fn_tip=h.selectbox("Tip condition", ["Adiabatic","Convective","Infinitely long"], key="fn_tip")
        elif calc == "Internal-flow convection (Nusselt)":
            a,b,c,d=st.columns(4); cv_re=a.number_input("Reynolds number", min_value=1.0, value=50000.0, key="cv_re"); cv_pr=b.number_input("Prandtl number", min_value=.01, value=7.0, key="cv_pr"); cv_d=c.number_input("Tube diameter (mm)", min_value=.001, value=25.0, key="cv_d"); cv_k=d.number_input("Fluid conductivity (W/m·K)", min_value=.0001, value=0.6, format="%.4f", key="cv_k")
            e,f,g=st.columns(3); cv_corr=e.selectbox("Turbulent correlation", ["Gnielinski","Dittus-Boelter"], key="cv_corr"); cv_heat=f.selectbox("Fluid is being", ["Heated","Cooled"], key="cv_heat"); cv_wall=g.selectbox("Laminar wall condition", ["Constant temperature","Constant heat flux"], key="cv_wall")
        elif calc == "Circular section properties":
            a,b=st.columns(2); cs_do=a.number_input("Outer diameter (mm)", min_value=.001, value=100.0, key="cs_do"); cs_di=b.number_input("Inner diameter (mm, 0 for solid)", min_value=0.0, value=80.0, key="cs_di")
        elif calc == "Radiation heat transfer":
            a,b,c,d=st.columns(4); rd_e=a.number_input("Emissivity", min_value=0.0, max_value=1.0, value=0.8, key="rd_e"); rd_a=b.number_input("Area (m²)", min_value=.0001, value=1.0, key="rd_a"); rd_ts=c.number_input("Surface temperature (K)", min_value=.1, value=400.0, key="rd_ts"); rd_tsur=d.number_input("Surroundings (K)", min_value=.1, value=300.0, key="rd_tsur")
        elif calc == "Damped vibration":
            a,b,c,d=st.columns(4); dv_m=a.number_input("Mass (kg)", min_value=.001, value=2.0, key="dv_m"); dv_k=b.number_input("Stiffness (N/m)", min_value=.001, value=20000.0, key="dv_k"); dv_c=c.number_input("Damping c (N·s/m)", min_value=0.0, value=40.0, key="dv_c"); dv_f=d.number_input("Forcing frequency (Hz, 0 to skip)", min_value=0.0, value=0.0, key="dv_f")
            e,f,g=st.columns(3); dv_x0=e.number_input("Initial displacement (mm)", value=10.0, key="dv_x0"); dv_v0=f.number_input("Initial velocity (m/s)", value=0.0, key="dv_v0"); dv_T=g.number_input("Plot duration (s)", min_value=.01, value=1.0, key="dv_T")
        elif calc == "Spur gear geometry":
            a,b,c,d=st.columns(4); sg_m=a.number_input("Module (mm)", min_value=.01, value=2.0, key="sg_m"); sg_n1=b.number_input("Pinion teeth", min_value=3, value=20, step=1, key="sg_n1"); sg_n2=c.number_input("Gear teeth", min_value=3, value=40, step=1, key="sg_n2"); sg_phi=d.number_input("Pressure angle (°)", min_value=10.0, max_value=30.0, value=20.0, key="sg_phi")
        elif calc == "Spur gear stress and sizing":
            a,b,c,d=st.columns(4); gs_p=a.number_input("Power (kW)", min_value=.001, value=5.0, key="gs_p"); gs_rpm=b.number_input("Pinion speed (RPM)", min_value=.001, value=1500.0, key="gs_rpm"); gs_m=c.number_input("Module (mm)", min_value=.01, value=3.0, key="gs_m"); gs_fw=d.number_input("Face width (mm)", min_value=.01, value=30.0, key="gs_fw")
            e,f,g,h=st.columns(4); gs_n1=e.number_input("Pinion teeth", min_value=12, value=20, step=1, key="gs_n1"); gs_n2=f.number_input("Gear teeth", min_value=12, value=40, step=1, key="gs_n2"); gs_fin=g.selectbox("Tooth finish", ["Hobbed","Milled","Ground"], key="gs_fin"); gs_ko=h.number_input("Application factor Ko", min_value=.5, value=1.25, key="gs_ko")
            i,j,k_=st.columns(3); gs_sb=i.number_input("Allowable bending stress (MPa)", min_value=.001, value=200.0, key="gs_sb"); gs_sc=j.number_input("Allowable contact stress (MPa)", min_value=.001, value=600.0, key="gs_sc"); gs_k=k_.number_input("Face width / module for sizing", min_value=1.0, value=10.0, key="gs_k")
        elif calc == "Rectangular section properties":
            a,b=st.columns(2); sec_b=a.number_input("Width b (mm)", min_value=.001, value=100.0); sec_h=b.number_input("Height h (mm)", min_value=.001, value=200.0)
        elif calc == "Beam deflection (point load)":
            a,b,c=st.columns(3); span=a.number_input("Span (m)", min_value=.001, value=4.0, key="defl_span"); load=b.number_input("Point load (N)", min_value=0.0, value=10000.0, key="defl_load"); pos=c.number_input("Load position from left (m)", min_value=0.0, value=2.0, key="defl_pos"); d,e=st.columns(2); e_gpa=d.number_input("Young's modulus E (GPa)", min_value=.001, value=200.0); i_cm4=e.number_input("Second moment I (cm⁴)", min_value=.001, value=2000.0)
        elif calc == "Euler column buckling":
            a,b,c,d=st.columns(4); e_gpa=a.number_input("Young's modulus E (GPa)", min_value=.001, value=200.0, key="buck_e"); i_cm4=b.number_input("Second moment I (cm⁴)", min_value=.001, value=100.0, key="buck_i"); col_len=c.number_input("Length (m)", min_value=.001, value=3.0); k_label=d.selectbox("End conditions (K)", ["Pinned–pinned (1.0)", "Fixed–fixed (0.5)", "Fixed–pinned (0.7)", "Fixed–free (2.0)"])
        elif calc == "Von Mises stress":
            a,b,c=st.columns(3); vsx=a.number_input("σx (MPa)", value=100.0, key="vm_sx"); vsy=b.number_input("σy (MPa)", value=50.0, key="vm_sy"); vtxy=c.number_input("τxy (MPa)", value=30.0, key="vm_txy")
        elif calc == "Solid shaft torsion":
            a,b,c=st.columns(3); tq=a.number_input("Torque (N·m)", min_value=.001, value=500.0); dia_mm=b.number_input("Diameter (mm)", min_value=.001, value=40.0); tau_allow=c.number_input("Allowable shear stress (MPa, for sizing)", min_value=.001, value=60.0)
        elif calc == "Thin-wall pressure vessel":
            a,b,c=st.columns(3); pv_p=a.number_input("Gauge pressure (MPa)", min_value=0.0, value=2.0); pv_r=b.number_input("Mean radius (mm)", min_value=.001, value=500.0); pv_t=c.number_input("Wall thickness (mm)", min_value=.001, value=10.0)
        elif calc == "Rectangular beam depth sizing":
            a,b,c,d,e=st.columns(5); load=a.number_input("Central load (N)", min_value=.001, value=10000.0); span=b.number_input("Span (m)", min_value=.001, value=3.0); width=c.number_input("Width (m)", min_value=.001, value=.1); allowable=d.number_input("Allowable stress (Pa)", min_value=1.0, value=150e6, format="%.1f"); cost=e.number_input("Material cost per m³", min_value=.001, value=1000.0)
        elif calc == "Mohr’s circle / principal stress":
            a,b,c=st.columns(3); sigx=a.number_input("σx normal stress (MPa)", value=80.0); sigy=b.number_input("σy normal stress (MPa)", value=20.0); tauxy=c.number_input("τxy shear stress (MPa)", value=15.0)
        elif calc == "Compression spring design":
            a,b,c=st.columns(3); cs_load=a.number_input("Axial load (N)", min_value=0.0, value=500.0, key="cs_load"); cs_wire=b.number_input("Wire diameter (mm)", min_value=.1, value=5.0, key="cs_wire"); cs_mean=c.number_input("Mean coil diameter (mm)", min_value=.2, value=40.0, key="cs_mean")
            d,e,f=st.columns(3); cs_coils=d.number_input("Active coils", min_value=1.0, value=8.0, key="cs_coils"); cs_g=e.number_input("Shear modulus (GPa)", min_value=.001, value=79.0, key="cs_g"); cs_free=f.number_input("Free length (mm; 0 to skip)", min_value=0.0, value=100.0, key="cs_free")
        elif calc == "Belt drive design":
            a,b,c=st.columns(3); bd_power=a.number_input("Transmitted power (W)", min_value=0.0, value=1500.0, key="bd_power"); bd_driver=b.number_input("Driver diameter (mm)", min_value=.1, value=200.0, key="bd_driver"); bd_driven=c.number_input("Driven diameter (mm)", min_value=.1, value=400.0, key="bd_driven")
            d,e,f=st.columns(3); bd_rpm=d.number_input("Driver speed (RPM)", min_value=.1, value=1440.0, key="bd_rpm"); bd_mu=e.number_input("Belt friction coefficient", min_value=0.0, value=.3, key="bd_mu"); bd_wrap=f.number_input("Wrap angle (degrees)", min_value=.1, max_value=360.0, value=170.0, key="bd_wrap")
        elif calc == "Power screw design":
            a,b,c=st.columns(3); ps_load=a.number_input("Axial load (N)", min_value=.1, value=10000.0, key="ps_load"); ps_d=b.number_input("Mean thread diameter (mm)", min_value=.1, value=30.0, key="ps_d"); ps_lead=c.number_input("Lead (mm/rev)", min_value=.01, value=5.0, key="ps_lead")
            d,e,f=st.columns(3); ps_mu=d.number_input("Thread friction coefficient", min_value=0.0, value=.15, key="ps_mu"); ps_muc=e.number_input("Collar friction coefficient", min_value=0.0, value=.08, key="ps_muc"); ps_dc=f.number_input("Collar mean diameter (mm)", min_value=0.0, value=40.0, key="ps_dc")
        elif calc == "Shaft design (fatigue)":
            a,b,c=st.columns(3); sf_m=a.number_input("Bending moment (N·m)", min_value=0.0, value=250.0, key="sf_m"); sf_tm=b.number_input("Mean torque (N·m)", min_value=0.0, value=100.0, key="sf_tm"); sf_ta=c.number_input("Alternating torque (N·m)", min_value=0.0, value=30.0, key="sf_ta")
            d,e,f=st.columns(3); sf_se=d.number_input("Endurance limit Se (MPa)", min_value=.1, value=200.0, key="sf_se"); sf_sut=e.number_input("Ultimate strength Sut (MPa)", min_value=.1, value=600.0, key="sf_sut"); sf_sy=f.number_input("Yield strength Sy (MPa)", min_value=.1, value=400.0, key="sf_sy")
            g,h,i=st.columns(3); sf_n=g.number_input("Target factor of safety", min_value=.1, value=2.0, key="sf_n"); sf_kfb=h.number_input("Bending fatigue factor Kf", min_value=1.0, value=1.5, key="sf_kfb"); sf_kft=i.number_input("Torsion fatigue factor Kfs", min_value=1.0, value=1.3, key="sf_kft")
        else:
            a,b,c=st.columns(3); speed=a.number_input("Launch speed (m/s)", min_value=.001, value=25.0); angle=b.slider("Launch angle (degrees)", 0.0, 90.0, 35.0); gravity=c.number_input("Gravity (m/s²)", min_value=.001, value=9.81)
        submitted=st.form_submit_button("Calculate", type="primary")
    if submitted:
        try:
            if calc == "Thermal expansion": result=et.thermal_expansion(L,alpha,dt)
            elif calc == "Shaft torque": result=et.shaft_torque(power,rpm)
            elif calc == "Spring natural frequency": result=et.spring_mass_frequency(k,mass)
            elif calc == "Factor of safety": result={"factor_of_safety":et.factor_of_safety(strength,stress)}
            elif calc == "Simply supported beam (point load)": result=et.simply_supported_beam_point_load(span,load,pos); st.plotly_chart(go.Figure([go.Scatter(x=result["x_m"],y=result["shear_n"],name="Shear force (N)"),go.Scatter(x=result["x_m"],y=result["moment_nm"],name="Bending moment (N·m)",yaxis="y2")]).update_layout(title="Beam shear and bending-moment diagrams",xaxis_title="Position (m)",yaxis=dict(title="Shear (N)"),yaxis2=dict(title="Moment (N·m)",overlaying="y",side="right")),use_container_width=True); result={k:v for k,v in result.items() if k not in ("x_m","shear_n","moment_nm")}
            elif calc == "Unit conversion": result={"converted_value":et.convert_units(val,from_unit,to_unit),"from":from_unit,"to":to_unit}
            elif calc == "Reynolds number": result={"reynolds_number":et.reynolds_number(rho,velocity,diameter,mu)}
            elif calc == "Pump power": result=et.pump_power(flow,dp,eta)
            elif calc == "Heat conduction": result={"heat_transfer_rate_w":et.conduction_heat_rate(conductivity,area,hot,cold,thickness)}
            elif calc == "Gear train": result=et.gear_train_output(rpm,int(driver),int(driven),torque,eta)
            elif calc == "Heat-engine performance": result=et.heat_engine_performance(qin,qout)
            elif calc == "Carnot efficiency": result={"carnot_efficiency":et.carnot_efficiency(th,tc),"carnot_efficiency_percent":100*et.carnot_efficiency(th,tc)}
            elif calc in ("Simply supported beam (uniform load)", "Cantilever beam (end load)", "Cantilever beam (uniform load)"):
                beam_fn={"Simply supported beam (uniform load)":et.simply_supported_beam_udl,"Cantilever beam (end load)":et.cantilever_beam_end_load,"Cantilever beam (uniform load)":et.cantilever_beam_udl}[calc]
                result=beam_fn(beam_len,beam_load,beam_e*1e9,beam_i*1e-8)
                st.plotly_chart(px.line(pd.DataFrame({"position_m":result["x_m"],"shear_n":result["shear_n"],"moment_nm":result["moment_nm"]}),x="position_m",y=["shear_n","moment_nm"],title="Shear force and bending moment (sagging positive)"),use_container_width=True)
                st.plotly_chart(px.line(pd.DataFrame({"position_m":result["x_m"],"deflection_mm":[-1000*v for v in result["deflection_m"]]}),x="position_m",y="deflection_mm",title="Deflected shape (downward negative, not to scale)"),use_container_width=True)
                result={k:v for k,v in result.items() if k not in ("x_m","shear_n","moment_nm","deflection_m")}
            elif calc in ("Multi-load beam", "Fixed–fixed beam", "Continuous beam"):
                if calc == "Continuous beam":
                    cb_end={"Pinned":"pinned","Fixed":"fixed","Free (overhang)":"free"}
                    result=et.continuous_beam([float(v) for v in cb_spans.replace(";",",").split(",") if v.strip()],et.parse_numeric_rows(cb_points,2),et.parse_numeric_rows(cb_udl,3),left_end=cb_end[cb_left],right_end=cb_end[cb_right],youngs_modulus_pa=cb_e*1e9,second_moment_m4=cb_i*1e-8,samples=601)
                elif calc == "Multi-load beam":
                    ml_kind={"Simply supported":"simply_supported","Cantilever (fixed at left end)":"cantilever","Fixed–fixed":"fixed_fixed","Propped cantilever (fixed left, pin right)":"propped_cantilever"}[ml_support]
                    result=et.beam_multi_load(ml_len,et.parse_numeric_rows(ml_points,2),et.parse_numeric_rows(ml_udl,3),ml_kind,ml_e*1e9,ml_i*1e-8,samples=401)
                elif ff_kind == "Point load":
                    result=et.fixed_fixed_beam_point_load(ff_len,ff_load,ff_pos,ff_e*1e9,ff_i*1e-8,samples=401)
                else:
                    result=et.fixed_fixed_beam_udl(ff_len,ff_load,ff_e*1e9,ff_i*1e-8,samples=401)
                st.plotly_chart(px.line(pd.DataFrame({"position_m":result["x_m"],"shear_n":result["shear_n"],"moment_nm":result["moment_nm"]}),x="position_m",y=["shear_n","moment_nm"],title="Shear force and bending moment (sagging positive)"),use_container_width=True)
                st.plotly_chart(px.line(pd.DataFrame({"position_m":result["x_m"],"deflection_mm":[-1000*v for v in result["deflection_m"]]}),x="position_m",y="deflection_mm",title="Deflected shape (downward negative, not to scale)"),use_container_width=True)
                result={k:v for k,v in result.items() if k not in ("x_m","shear_n","moment_nm","deflection_m")}
            elif calc == "Fatigue (modified Goodman)":
                raw=et.goodman_fatigue(fg_sut*1e6,fg_se*1e6,stress_max_pa=fg_max*1e6,stress_min_pa=fg_min*1e6,yield_strength_pa=fg_sy*1e6 if fg_sy>0 else None)
                result={(k[:-3]+"_mpa" if k.endswith("_pa") else k):(v/1e6 if k.endswith("_pa") else v) for k,v in raw.items()}
                fig=go.Figure([go.Scatter(x=[0,fg_sut],y=[fg_se,0],mode="lines",name="Goodman line"),go.Scatter(x=[-fg_sut,0],y=[fg_se,fg_se],mode="lines",name="Compressive-mean limit",line=dict(dash="dot"))])
                if fg_sy>0: fig.add_trace(go.Scatter(x=[0,fg_sy],y=[fg_sy,0],mode="lines",name="Yield (Langer) line",line=dict(dash="dash")))
                fig.add_trace(go.Scatter(x=[result["stress_mean_mpa"]],y=[result["stress_amplitude_mpa"]],mode="markers",name="Operating point",marker=dict(size=12,symbol="x")))
                st.plotly_chart(fig.update_layout(title="Goodman diagram",xaxis_title="Mean stress (MPa)",yaxis_title="Alternating stress (MPa)",yaxis_range=[0,1.1*fg_se if fg_sy<=0 else 1.1*max(fg_se,fg_sy)]),use_container_width=True)
                result={k:("infinite (static load)" if isinstance(v,float) and math.isinf(v) else v) for k,v in result.items()}
            elif calc == "Stress concentration (Kt, Kf)":
                nominal=sc_nom*1e6 if sc_nom>0 else None
                if sc_geo.startswith("Circular"): raw=et.stress_concentration_factor("circular_hole_finite_plate",width_m=sc_w/1000,hole_diameter_m=sc_d/1000,nominal_stress_pa=nominal)
                elif sc_geo.startswith("Elliptical"): raw=et.stress_concentration_factor("elliptical_hole_infinite_plate",semi_axis_transverse_m=sc_a/1000,semi_axis_parallel_m=sc_b/1000,nominal_stress_pa=nominal)
                else:
                    raw={"geometry":"user_supplied","kt":sc_kt}
                    if nominal is not None: raw.update({"nominal_stress_pa":nominal,"peak_stress_pa":sc_kt*nominal})
                result={(k[:-3]+"_mpa" if k.endswith("_pa") else k):(v/1e6 if k.endswith("_pa") else v) for k,v in raw.items()}
                if sc_r>0:
                    kfr=et.fatigue_notch_factor(raw["kt"],sc_r/1000,sc_sut*1e6,loading=sc_load.lower(),method=sc_method.lower())
                    result.update({"notch_sensitivity_q":kfr["q"],"fatigue_notch_factor_kf":kfr["kf"],"notch_sensitivity_model":kfr["method"]})
                    if nominal is not None: result["fatigue_peak_stress_mpa"]=kfr["kf"]*sc_nom
                if sc_geo.startswith("Circular"):
                    ratios=np.linspace(0.01,0.65,60); kts=[et.stress_concentration_factor("circular_hole_finite_plate",width_m=1.0,hole_diameter_m=float(x))["kt"] for x in ratios]
                    fig=go.Figure([go.Scatter(x=ratios,y=kts,mode="lines",name="Kt (net section)")]).add_trace(go.Scatter(x=[raw["diameter_to_width_ratio"]],y=[raw["kt"]],mode="markers",name="This plate",marker=dict(size=12,symbol="x")))
                    st.plotly_chart(fig.update_layout(title="Kt for a central hole in a finite-width plate",xaxis_title="d / w",yaxis_title="Kt"),use_container_width=True)
            elif calc == "S-N fatigue life":
                raw=et.sn_fatigue_life(sn_sut*1e6,sn_se*1e6,stress_max_pa=sn_max*1e6,stress_min_pa=sn_min*1e6,notch_factor_kf=sn_kf,fatigue_strength_fraction=sn_f,target_cycles=sn_target if sn_target>0 else None)
                result={(k[:-3]+"_mpa" if k.endswith("_pa") else k):(v/1e6 if k.endswith("_pa") else v) for k,v in raw.items()}
                curve=et.sn_curve(sn_sut*1e6,sn_se*1e6,fatigue_strength_fraction=sn_f,points=80,max_cycles=1e7)
                fig=go.Figure([go.Scatter(x=curve["cycles"],y=[v/1e6 for v in curve["strength_pa"]],mode="lines",name="S-N line")])
                fig.add_hline(y=raw["equivalent_reversed_stress_pa"]/1e6,line_dash="dash",annotation_text="Equivalent reversed stress")
                if not raw["infinite_life_predicted"] and raw["life_cycles"]>=1e3: fig.add_trace(go.Scatter(x=[raw["life_cycles"]],y=[raw["equivalent_reversed_stress_pa"]/1e6],mode="markers",name="Predicted life",marker=dict(size=12,symbol="x")))
                st.plotly_chart(fig.update_layout(title="S-N curve (high-cycle range)",xaxis_title="Cycles to failure N",yaxis_title="Fully reversed strength (MPa)",xaxis_type="log",yaxis_type="log"),use_container_width=True)
                result={k:("infinite" if isinstance(v,float) and math.isinf(v) else v) for k,v in result.items()}
            elif calc == "Heat exchanger (LMTD)":
                hx_kind="parallel_flow" if hx_arr == "Parallel flow" else "counterflow"
                hx_f=et.shell_and_tube_correction_factor(hx_thi,hx_tho,hx_tci,hx_tco) if hx_arr.startswith("Shell") else 1.0
                hx_kw={"Heat duty (from U and area)":{"overall_u_w_m2k":hx_u,"area_m2":hx_area},"Required area (from U and duty)":{"overall_u_w_m2k":hx_u,"heat_duty_w":hx_q}}.get(hx_mode,{})
                result=et.heat_exchanger_lmtd(hx_thi,hx_tho,hx_tci,hx_tco,arrangement=hx_kind,correction_factor=hx_f,**hx_kw)
            elif calc == "Heat exchanger (effectiveness-NTU)":
                nt_kind=NTU_LABELS[nt_arr]
                if nt_mode.startswith("Rating"): result=et.heat_exchanger_ntu(nt_ch,nt_cc,nt_thi,nt_tci,ua_w_k=nt_ua,arrangement=nt_kind)
                else: result=et.heat_exchanger_ntu(nt_ch,nt_cc,nt_thi,nt_tci,effectiveness=nt_eps,arrangement=nt_kind)
                ntu_axis=np.linspace(0,max(5.0,1.3*result["ntu"]),120)
                fig=go.Figure(go.Scatter(x=ntu_axis,y=[et.exchanger_effectiveness(float(n),result["capacity_ratio"],nt_kind) for n in ntu_axis],mode="lines",name=f"Cr = {result['capacity_ratio']:.3g}"))
                fig.add_trace(go.Scatter(x=[result["ntu"]],y=[result["effectiveness"]],mode="markers",name="This design",marker=dict(size=12)))
                st.plotly_chart(fig.update_layout(title="Effectiveness versus NTU",xaxis_title="NTU",yaxis_title="Effectiveness"),use_container_width=True)
            elif calc == "Pipe flow pressure drop":
                flow_kw={"flow_rate_m3_s":pp_flow/1000} if pp_mode.startswith("Flow") else {"velocity_m_s":pp_flow}
                result=et.pipe_pressure_drop(pp_d/1000,pp_len,pp_rho,pp_mu/1000,roughness_m=pp_rough/1000,minor_loss_k=pp_k,elevation_gain_m=pp_dz,**flow_kw)
                vs=np.linspace(0.05,max(3.0,2*result["velocity_m_s"]),60)
                dps=[et.pipe_pressure_drop(pp_d/1000,pp_len,pp_rho,pp_mu/1000,velocity_m_s=float(v),roughness_m=pp_rough/1000,minor_loss_k=pp_k,elevation_gain_m=pp_dz)["total_pressure_drop_pa"]/1000 for v in vs]
                fig=go.Figure(go.Scatter(x=vs,y=dps,mode="lines",name="System curve"))
                fig.add_trace(go.Scatter(x=[result["velocity_m_s"]],y=[result["total_pressure_drop_pa"]/1000],mode="markers",name="Operating point",marker=dict(size=12)))
                st.plotly_chart(fig.update_layout(title="Pressure drop versus velocity",xaxis_title="Velocity (m/s)",yaxis_title="Total pressure drop (kPa)"),use_container_width=True)
            elif calc == "Rolling-bearing life (L10)":
                p_eq=et.bearing_equivalent_load(br_fr*1000,br_fa*1000,radial_factor_x=br_x,axial_factor_y=br_y)
                result={"equivalent_load_n":p_eq,**et.bearing_life_l10(br_c*1000,p_eq,speed_rpm=br_rpm,bearing_type=br_type.lower(),reliability_percent=br_rel,target_life_hours=br_target if br_target>0 else None)}
            elif calc == "Bolted joint":
                at=et.metric_tensile_stress_area(bj_d/1000,bj_p/1000)
                raw=et.bolted_joint(bj_d/1000,at,bj_sp*1e6,bj_c,external_load_n=bj_load*1000,preload_fraction=bj_frac,torque_coefficient_k=bj_k)
                result={"tensile_stress_area_mm2":at*1e6,**raw}
                result={k:("infinite" if isinstance(v,float) and math.isinf(v) else v) for k,v in result.items()}
                pl=np.linspace(0,max(2*bj_load,1.0)*1000,60)
                fb=[raw["preload_n"]+bj_c*p for p in pl]; fm=[raw["preload_n"]-(1-bj_c)*p for p in pl]
                fig=go.Figure([go.Scatter(x=pl/1000,y=[v/1000 for v in fb],name="Bolt load"),go.Scatter(x=pl/1000,y=[v/1000 for v in fm],name="Clamp load")])
                fig.add_hline(y=raw["proof_load_n"]/1000,line_dash="dash",annotation_text="Proof load")
                st.plotly_chart(fig.update_layout(title="Bolt and clamp load versus external load",xaxis_title="External load (kN)",yaxis_title="Load (kN)"),use_container_width=True)
            elif calc == "Straight fin":
                tip_kind={"Adiabatic":"adiabatic","Convective":"convective","Infinitely long":"infinite"}[fn_tip]
                result=et.straight_fin(fn_k,fn_h,fn_len/1000,fn_t/1000,fn_w/1000,fn_tb,fn_ta,tip=tip_kind)
                m_fin=result["fin_parameter_m_per_m"]; L_fin=fn_len/1000; xs=np.linspace(0,L_fin,100); th=fn_tb-fn_ta
                if tip_kind=="infinite": prof=th*np.exp(-m_fin*xs)
                elif tip_kind=="adiabatic": prof=th*np.cosh(m_fin*(L_fin-xs))/np.cosh(m_fin*L_fin)
                else:
                    bt=fn_h/(m_fin*fn_k); prof=th*(np.cosh(m_fin*(L_fin-xs))+bt*np.sinh(m_fin*(L_fin-xs)))/(np.cosh(m_fin*L_fin)+bt*np.sinh(m_fin*L_fin))
                st.plotly_chart(px.line(pd.DataFrame({"distance_from_base_mm":xs*1000,"temperature_c":fn_ta+prof}),x="distance_from_base_mm",y="temperature_c",title="Fin temperature profile"),use_container_width=True)
            elif calc == "Internal-flow convection (Nusselt)":
                result=et.convection_internal_flow(cv_re,cv_pr,cv_d/1000,cv_k,correlation=cv_corr.lower().replace("-","_"),heating=cv_heat=="Heated",wall_condition=cv_wall.lower().replace(" ","_"))
            elif calc == "Circular section properties": result=et.circular_section_properties(cs_do/1000,cs_di/1000)
            elif calc == "Radiation heat transfer": result=et.radiation_heat_rate(rd_e,rd_a,rd_ts,rd_tsur)
            elif calc == "Damped vibration":
                raw=et.damped_vibration(dv_m,dv_k,dv_c,forcing_frequency_hz=dv_f if dv_f>0 else None,initial_displacement_m=dv_x0/1000,initial_velocity_m_s=dv_v0,response_duration_s=dv_T)
                st.plotly_chart(px.line(pd.DataFrame({"time_s":raw["time_s"],"displacement_mm":[1000*v for v in raw["displacement_m"]]}),x="time_s",y="displacement_mm",title=f"Free response ({raw['damping_type']})"),use_container_width=True)
                result={k:("infinite" if isinstance(v,float) and math.isinf(v) else v) for k,v in raw.items() if k not in ("time_s","displacement_m")}
                zeta=raw["damping_ratio"]; rr=np.linspace(0.01,3,200)
                tr=[et.damped_vibration(dv_m,dv_k,dv_c,forcing_frequency_hz=float(r_*raw["natural_frequency_hz"]))["transmissibility"] for r_ in rr]
                st.plotly_chart(px.line(pd.DataFrame({"frequency_ratio":rr,"transmissibility":[min(v,10) for v in tr]}),x="frequency_ratio",y="transmissibility",title=f"Transmissibility versus frequency ratio (ζ = {zeta:.3g}, capped at 10)"),use_container_width=True)
            elif calc == "Spur gear geometry": result=et.spur_gear_geometry(sg_m/1000,int(sg_n1),int(sg_n2),pressure_angle_deg=sg_phi)
            elif calc == "Spur gear stress and sizing":
                result=et.spur_gear_stress(gs_p*1000,gs_rpm,gs_m/1000,int(gs_n1),int(gs_n2),gs_fw/1000,finish=gs_fin.lower(),application_factor=gs_ko,allowable_bending_pa=gs_sb*1e6,allowable_contact_pa=gs_sc*1e6)
                result["minimum_module_for_bending"]=et.spur_gear_module_for_bending(gs_p*1000,gs_rpm,int(gs_n1),gs_sb*1e6,face_width_to_module=gs_k,finish=gs_fin.lower(),application_factor=gs_ko)
                ms=np.linspace(max(gs_m/2,.5),gs_m*2,40)
                sbs=[et.spur_gear_stress(gs_p*1000,gs_rpm,mm/1000,int(gs_n1),int(gs_n2),gs_k*mm/1000,finish=gs_fin.lower(),application_factor=gs_ko)["bending_stress_pinion_pa"]/1e6 for mm in ms]
                fig=px.line(pd.DataFrame({"module_mm":ms,"pinion_bending_stress_mpa":sbs}),x="module_mm",y="pinion_bending_stress_mpa",title=f"Pinion bending stress versus module (face width = {gs_k:g} m)")
                fig.add_hline(y=gs_sb,line_dash="dash",annotation_text="Allowable")
                st.plotly_chart(fig,use_container_width=True)
            elif calc == "Rectangular section properties": result=et.rectangular_section_properties(sec_b/1000,sec_h/1000)
            elif calc == "Beam deflection (point load)":
                result=et.beam_deflection_point_load(span,load,pos,e_gpa*1e9,i_cm4*1e-8)
                st.plotly_chart(px.line(pd.DataFrame({"position_m":result["x_m"],"deflection_mm":[-1000*v for v in result["deflection_m"]]}),x="position_m",y="deflection_mm",title="Beam deflected shape (downward negative, not to scale)"),use_container_width=True)
                result={k:v for k,v in result.items() if k not in ("x_m","deflection_m")}
            elif calc == "Euler column buckling": result=et.euler_buckling_load(e_gpa*1e9,i_cm4*1e-8,col_len,float(k_label.split("(")[1].rstrip(")")))
            elif calc == "Von Mises stress": result={"von_mises_stress_mpa":et.von_mises_stress(vsx,vsy,vtxy)}
            elif calc == "Solid shaft torsion": result={**et.torsion_solid_shaft(tq,dia_mm/1000),"minimum_diameter_for_allowable_mm":1000*et.shaft_diameter_for_torque(tq,tau_allow*1e6)}
            elif calc == "Thin-wall pressure vessel": result=et.thin_wall_pressure_vessel(pv_p*1e6,pv_r/1000,pv_t/1000)
            elif calc == "Rectangular beam depth sizing": result=optimize_beam_depth(load,span,width,allowable,cost)
            elif calc == "Compression spring design":
                raw=et.compression_spring_design(cs_load,cs_wire/1000,cs_mean/1000,cs_coils,cs_g*1e9,free_length_m=cs_free/1000 if cs_free>0 else None)
                result={k:(v*1000 if k in ("deflection_m","solid_length_m","remaining_length_at_load_m","estimated_solid_clearance_m") else v/1e6 if k.endswith("_pa") else v) for k,v in raw.items()}
                result["units_note"]="Lengths in mm, stress in MPa, spring rate in N/m."
            elif calc == "Belt drive design":
                result=et.belt_drive_design(bd_power,bd_driver/1000,bd_driven/1000,bd_rpm,bd_mu,math.radians(bd_wrap))
            elif calc == "Power screw design":
                result=et.power_screw_design(ps_load,ps_d/1000,ps_lead/1000,ps_mu,collar_friction_coefficient=ps_muc,collar_mean_diameter_m=ps_dc/1000)
            elif calc == "Shaft design (fatigue)":
                raw=et.shaft_fatigue_design(sf_m,sf_tm,sf_ta,sf_se*1e6,sf_sut*1e6,sf_sy*1e6,factor_of_safety=sf_n,kf_bending=sf_kfb,kf_torsion=sf_kft)
                result=raw
            elif calc == "Mohr’s circle / principal stress":
                circle=MohrsCircle(sigx,sigy,tauxy); result={"average_normal_stress_mpa":circle.circleCenter(),"radius_max_shear_mpa":circle.radius(),"maximum_principal_stress_mpa":circle.maximumPrincipalStress(),"minimum_principal_stress_mpa":circle.minimumPrincipalStress(),"maximum_shear_stress_mpa":circle.maximumShearStress(),"principal_angle_degrees":circle.principalAngle()}
                theta=np.linspace(0,2*np.pi,400); st.plotly_chart(go.Figure(go.Scatter(x=circle.circleCenter()+circle.radius()*np.cos(theta),y=circle.radius()*np.sin(theta),mode="lines",name="Mohr circle")).add_trace(go.Scatter(x=[sigx,sigy],y=[tauxy,-tauxy],mode="markers+lines",name="Stress states")).update_layout(title="Mohr’s circle",xaxis_title="Normal stress (MPa)",yaxis_title="Shear stress (MPa)"),use_container_width=True)
            else:
                t,x,y=projectile_trajectory(speed,angle,gravity=gravity); result={"flight_time_s":float(t[-1]),"range_m":float(x[-1]),"maximum_height_m":float(np.max(y))}; st.plotly_chart(px.line(pd.DataFrame({"horizontal_distance_m":x,"height_m":y}),x="horizontal_distance_m",y="height_m",title="Projectile trajectory (no air resistance)"),use_container_width=True)
            show_result(calc,result)
        except Exception as exc: st.error(f"Calculation failed: {exc}")

with tabs[2]:
    st.subheader("CSV import and interactive data explorer")
    uploaded=st.file_uploader("Import sensor / test / machine CSV", type=["csv"], key="data_csv")
    if uploaded is not None:
        try:
            st.session_state.dataset=pd.read_csv(uploaded)
            st.session_state.dataset_name=uploaded.name
        except Exception as exc: st.error(f"Could not read CSV: {exc}")
    df=st.session_state.dataset
    if df is not None and not df.empty:
        st.caption(f"Loaded: {getattr(st.session_state,'dataset_name','dataset.csv')} • {len(df):,} rows × {len(df.columns)} columns")
        st.dataframe(df.head(1000),use_container_width=True)
        ncols=df.select_dtypes(include="number").columns.tolist()
        c1,c2,c3=st.columns(3); c1.metric("Rows",f"{len(df):,}"); c2.metric("Columns",len(df.columns)); c3.metric("Missing cells",int(df.isna().sum().sum()))
        if ncols:
            chosen=st.multiselect("Numeric columns to graph",ncols,default=ncols[:min(3,len(ncols))])
            xcol=st.selectbox("X-axis (optional row order)",["(row index)"]+df.columns.tolist(),index=0)
            plot_df=df.copy().reset_index(names="row_index")
            xaxis="row_index" if xcol=="(row index)" else xcol
            if chosen:
                fig=px.line(plot_df,x=xaxis,y=chosen,title="Interactive measurements — zoom, hover and toggle series")
                st.plotly_chart(fig,use_container_width=True)
            st.subheader("Descriptive statistics")
            st.dataframe(df[ncols].describe().T,use_container_width=True)
            with st.expander("Correlation heatmap"):
                corr=df[ncols].corr(numeric_only=True)
                st.plotly_chart(px.imshow(corr,text_auto=".2f",aspect="auto",title="Numeric feature correlation"),use_container_width=True)
            st.subheader("Outlier detection")
            anomaly_col=st.selectbox("Measurement column",ncols,key="anomaly_col")
            threshold=st.slider("Robust anomaly threshold",1.0,6.0,3.5,.5)
            try:
                valid=df[anomaly_col].dropna(); flags=detect_anomalies(valid.to_numpy(),z_threshold=threshold)
                anomalies=pd.DataFrame({"row_index":valid.index,"value":valid.values,"is_anomaly":flags})
                st.metric("Flagged points",int(flags.sum()))
                st.plotly_chart(px.scatter(anomalies,x="row_index",y="value",color="is_anomaly",title="Robust outlier flags"),use_container_width=True)
                st.download_button("Download outlier results CSV",anomalies.to_csv(index=False).encode(),"outlier_results.csv","text/csv")
                st.session_state.latest_data_report={"file":getattr(st.session_state,'dataset_name','dataset.csv'),"rows":len(df),"columns":len(df.columns),"missing_cells":int(df.isna().sum().sum()),"summary":eda_summary(df),"anomalies":int(flags.sum()),"anomaly_column":anomaly_col}
            except Exception as exc: st.error(f"Outlier analysis: {exc}")
            st.subheader("Baseline forecast")
            fc1,fc2=st.columns(2); forecast_col=fc1.selectbox("Column to forecast",ncols,key="forecast_col"); horizon=fc2.number_input("Forecast horizon (samples)",1,200,10); window=st.slider("Moving-average window",1,50,5)
            series=df[forecast_col].dropna().to_numpy(dtype=float)
            if len(series)>=1:
                pred=moving_average_forecast(series,int(horizon),int(window))
                fdf=pd.DataFrame({"sample":list(range(len(series)))+list(range(len(series),len(series)+len(pred))),"value":list(series)+[np.nan]*len(pred),"forecast": [np.nan]*len(series)+list(pred)})
                st.plotly_chart(px.line(fdf,x="sample",y=["value","forecast"],title="Historical data and moving-average baseline"),use_container_width=True)
                st.download_button("Download forecast CSV",pd.DataFrame({"forecast_step":np.arange(1,len(pred)+1),"forecast":pred}).to_csv(index=False).encode(),"forecast.csv","text/csv")
        cleaned=df.copy()
        clean_mode=st.selectbox("Missing-data handling for export",["Keep as-is","Drop rows with missing values","Fill numeric missing values with median"])
        if clean_mode=="Drop rows with missing values": cleaned=cleaned.dropna()
        elif clean_mode=="Fill numeric missing values with median":
            for col in cleaned.select_dtypes(include="number").columns: cleaned[col]=cleaned[col].fillna(cleaned[col].median())
        st.download_button("Download processed CSV",cleaned.to_csv(index=False).encode(),"processed_data.csv","text/csv")
    else:
        st.info("Upload a CSV to unlock live plots, statistics, correlations, anomaly detection, forecasting and processed-data export.")
        demo=pd.DataFrame({"time_s":np.arange(100),"vibration_mm_s":np.sin(np.arange(100)/5)+np.random.default_rng(3).normal(0,.1,100),"temperature_c":60+np.arange(100)*.05+np.random.default_rng(4).normal(0,.3,100)})
        st.plotly_chart(px.line(demo,x="time_s",y=["vibration_mm_s","temperature_c"],title="Demo data preview (upload a CSV to use your own data)"),use_container_width=True)

with tabs[3]:
    st.subheader("Train a supervised regression model")
    st.write("Select a numeric target and predictors. Data is split into training and test sets; preprocessing is fitted on training data only.")
    ml_file=st.file_uploader("Upload training CSV (or use CSV already imported above)",type=["csv"],key="ml_csv")
    mldf=None
    if ml_file is not None:
        try: mldf=pd.read_csv(ml_file)
        except Exception as exc: st.error(str(exc))
    elif st.session_state.dataset is not None: mldf=st.session_state.dataset.copy()
    if mldf is not None:
        numeric_cols=mldf.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols)<2: st.warning("ML regression needs at least two numeric columns: one target and one or more features.")
        else:
            target=st.selectbox("Target to predict",numeric_cols,index=len(numeric_cols)-1)
            feature_options=[c for c in numeric_cols if c!=target]
            features=st.multiselect("Input features",feature_options,default=feature_options)
            c1,c2,c3=st.columns(3); model_name=c1.selectbox("Model",["Random Forest Regressor","Linear Regression","Gradient Boosting Regressor"]); test_size=c2.slider("Test-set fraction",.1,.4,.2,.05); seed=c3.number_input("Random seed",0,10000,42)
            if st.button("Train model",type="primary",disabled=not features):
                try:
                    from sklearn.compose import ColumnTransformer
                    from sklearn.impute import SimpleImputer
                    from sklearn.pipeline import Pipeline
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                    from sklearn.linear_model import LinearRegression
                    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                    clean=mldf[features+[target]].replace([np.inf,-np.inf],np.nan).dropna(subset=[target])
                    if len(clean)<10: raise ValueError("At least 10 rows with a valid target are required.")
                    X=clean[features]; y=clean[target]
                    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=float(test_size),random_state=int(seed))
                    numeric_transform=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
                    pre=ColumnTransformer([("numeric",numeric_transform,features)])
                    estimator={"Random Forest Regressor":RandomForestRegressor(n_estimators=200,random_state=int(seed),n_jobs=-1),"Linear Regression":LinearRegression(),"Gradient Boosting Regressor":GradientBoostingRegressor(random_state=int(seed))}[model_name]
                    model=Pipeline([("preprocess",pre),("model",estimator)])
                    model.fit(X_train,y_train); y_pred=model.predict(X_test)
                    metrics={"model":model_name,"target":target,"features":features,"training_rows":len(X_train),"test_rows":len(X_test),"mae":float(mean_absolute_error(y_test,y_pred)),"rmse":float(np.sqrt(mean_squared_error(y_test,y_pred))),"r2":float(r2_score(y_test,y_pred))}
                    st.session_state.latest_ml={"model":model,"metrics":metrics,"actual":np.asarray(y_test),"predicted":np.asarray(y_pred),"features":features,"target":target,"model_name":model_name,"source_rows":len(clean)}
                    record_activity("ml_training",metrics)
                except Exception as exc: st.error(f"Training failed: {exc}")
            trained=st.session_state.latest_ml
            if trained:
                metrics=trained["metrics"]
                st.markdown(f"**Latest trained model:** {metrics['model']} → target `{metrics['target']}`")
                a,b,c=st.columns(3); a.metric("MAE",f"{metrics['mae']:.5g}"); b.metric("RMSE",f"{metrics['rmse']:.5g}"); c.metric("R²",f"{metrics['r2']:.4f}")
                actual_pred=pd.DataFrame({"Actual":trained["actual"],"Predicted":trained["predicted"]})
                st.plotly_chart(px.scatter(actual_pred,x="Actual",y="Predicted",title="Test set: actual vs predicted"),use_container_width=True)
                st.dataframe(actual_pred,use_container_width=True)
                st.download_button("Download test predictions CSV",actual_pred.to_csv(index=False).encode(),"ml_test_predictions.csv","text/csv")
                st.download_button("Download trained model (pickle)",pickle.dumps(trained["model"]),"trained_mechanical_model.pkl","application/octet-stream",help="Only load pickle files you created or trust; pickle can execute code when loaded.")
                st.session_state.latest_ml_report=metrics
                st.caption("Model quality depends on dataset quality, leakage, sample size and the train/test split. This is a regression workflow, not a guarantee of real-world performance.")
    else: st.info("Import a numeric CSV in this tab or first load one in Live Graphs & CSV.")

with tabs[4]:
    st.subheader("Interactive physical simulations")
    sim=st.radio("Simulation",["Mass–spring–damper vibration","Beam sizing sensitivity"],horizontal=True)
    if sim=="Mass–spring–damper vibration":
        a,b,c=st.columns(3); m=a.number_input("Mass (kg)",min_value=.01,value=10.0); k=b.number_input("Stiffness (N/m)",min_value=.1,value=1000.0); damp=c.number_input("Damping (N·s/m)",min_value=0.0,value=5.0)
        d,e,f=st.columns(3); x0=d.number_input("Initial displacement (m)",value=.05); v0=e.number_input("Initial velocity (m/s)",value=0.0); duration=f.slider("Duration (s)",1.0,60.0,10.0)
        try:
            simres=simulate_mass_spring(m,k,damp,x0,v0,duration_s=float(duration),samples=1201)
            simdf=pd.DataFrame(simres)
            st.plotly_chart(px.line(simdf,x="time_s",y=["displacement_m","velocity_m_s"],title="Live mass–spring–damper response"),use_container_width=True)
            c1,c2,c3=st.columns(3); c1.metric("Natural frequency",f"{math.sqrt(k/m)/(2*math.pi):.3f} Hz"); c2.metric("Damping ratio",f"{damp/(2*math.sqrt(k*m)):.4f}"); c3.metric("Peak |displacement|",f"{np.max(np.abs(simres['displacement_m'])):.5g} m")
            st.download_button("Download simulation CSV",simdf.to_csv(index=False).encode(),"vibration_simulation.csv","text/csv")
            if st.button("Save vibration simulation"): record_activity("mass_spring_simulation",{"mass_kg":m,"stiffness_n_m":k,"damping_n_s_m":damp,"duration_s":duration,"peak_displacement_m":float(np.max(np.abs(simres['displacement_m'])))})
        except Exception as exc: st.error(str(exc))
    else:
        a,b,c=st.columns(3); load=a.number_input("Load (N)",min_value=1.0,value=10000.0,key="sens_load"); span=b.number_input("Span (m)",min_value=.1,value=3.0,key="sens_span"); width=c.number_input("Width (m)",min_value=.01,value=.1,key="sens_width")
        allowable=st.number_input("Allowable stress (MPa)",min_value=.1,value=150.0)
        depths=np.linspace(.02,.8,100); stresses=3*load*span/(2*width*depths**2); required=optimize_beam_depth(load,span,width,allowable*1e6)
        sdf=pd.DataFrame({"depth_m":depths,"bending_stress_mpa":stresses/1e6})
        fig=px.line(sdf,x="depth_m",y="bending_stress_mpa",title="Beam depth vs bending stress"); fig.add_hline(y=allowable,line_dash="dash",annotation_text="Allowable stress")
        fig.add_vline(x=required["depth_m"],line_dash="dot",annotation_text="Required depth")
        st.plotly_chart(fig,use_container_width=True); st.json(required)
        if st.button("Save beam sizing"): record_activity("beam_depth_sizing",required)

with tabs[5]:
    st.subheader("Create a portable PDF report")
    include_history=st.checkbox("Include recent session activity",value=True)
    sections=[]
    sections.append(("Dashboard summary",{"Generated by":"Mechanical Engineering All-in-One Dashboard","Session calculations":len(st.session_state.activity_log),"CSV loaded":getattr(st.session_state,"dataset_name","No CSV loaded"),"ML model trained":"Yes" if st.session_state.latest_ml else "No"}))
    if st.session_state.activity_log:
        recent=pd.DataFrame([{"Time":x["time"],"Activity":x["kind"],"Result":json.dumps(x["result"],ensure_ascii=False)[:180]} for x in st.session_state.activity_log[-20:]])
        if include_history: sections.append(("Recent session activity",recent))
        sections.append(("Latest calculation / activity",st.session_state.activity_log[-1]["result"]))
    if st.session_state.dataset is not None:
        sections.append(("Imported CSV overview",{"File":getattr(st.session_state,"dataset_name","dataset.csv"),"Rows":len(st.session_state.dataset),"Columns":len(st.session_state.dataset.columns),"Missing cells":int(st.session_state.dataset.isna().sum().sum())}))
        sections.append(("CSV numeric summary",st.session_state.dataset.select_dtypes(include="number").describe().T.reset_index(names="column")))
    if st.session_state.latest_ml:
        sections.append(("Machine-learning evaluation",st.session_state.latest_ml["metrics"]))
    pdf_title=st.text_input("Report title","Mechanical Engineering Analysis Report")
    if st.button("Generate PDF report",type="primary"):
        try:
            pdf=make_pdf(pdf_title,sections)
            st.session_state.latest_pdf=pdf
            record_activity("pdf_report",{"title":pdf_title,"bytes":len(pdf)})
            st.success("PDF report generated.")
        except Exception as exc: st.error(f"PDF generation failed: {exc}. Ensure reportlab is installed.")
    if st.session_state.get("latest_pdf"):
        st.download_button("Download PDF report",st.session_state.latest_pdf,"mechanical_engineering_report.pdf","application/pdf")
    st.divider(); st.subheader("Saved SQLite history")
    try:
        saved=store.list(limit=100)
        if saved:
            history_df=pd.DataFrame([{"ID":r["id"],"Time":datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M:%S"),"Type":r["kind"],"Payload":json.dumps(r["payload"],ensure_ascii=False)[:200]} for r in saved])
            st.dataframe(history_df,use_container_width=True)
            st.download_button("Download saved history CSV",history_df.to_csv(index=False).encode(),"engineering_history.csv","text/csv")
        else: st.caption("No saved records yet. Run a calculation or simulation to populate history.")
    except Exception as exc: st.caption(f"History store unavailable: {exc}")
