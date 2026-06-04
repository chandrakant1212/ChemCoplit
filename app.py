import os

# ── Prevent transformers from importing TensorFlow ──
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import streamlit as st
import json
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from rag.retriever import load_vectorstore
from rag.pipeline import get_engineering_response
from calculations.fluidization import calc_umf_wen_yu, calc_terminal_velocity, calc_bubble_velocity
from calculations.heat_exchanger import calc_lmtd, calc_heat_exchanger_area
from calculations.reactor_design import cstr_volume, pfr_volume_nth_order

load_dotenv()


st.set_page_config(
    page_title="ChemCopilot",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header banner */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid #e94560;
        box-shadow: 0 8px 32px rgba(233, 69, 96, 0.15);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
    }

    /* Calculation result card */
    .calc-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        transition: border-color 0.3s ease;
    }
    .calc-card:hover {
        border-color: #e94560;
    }

    /* Result highlight strip */
    .result-highlight {
        background: linear-gradient(90deg, #0f3460 0%, #16213e 100%);
        border-left: 4px solid #e94560;
        padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.92rem;
        color: #e2e8f0;
        margin: 0.5rem 0;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .status-ok {
        background: rgba(72, 187, 120, 0.15);
        color: #48bb78;
        border: 1px solid rgba(72, 187, 120, 0.3);
    }
    .status-warn {
        background: rgba(237, 137, 54, 0.15);
        color: #ed8936;
        border: 1px solid rgba(237, 137, 54, 0.3);
    }

    /* Sidebar branding */
    .sidebar-brand {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
    }
    .sidebar-brand h2 {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #e94560, #f39c12);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-brand p {
        font-size: 0.85rem;
        color: #a0aec0;
        margin: 0.2rem 0 0 0;
    }

    /* Example query buttons */
    .example-btn {
        display: block;
        width: 100%;
        text-align: left;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        background: rgba(15, 52, 96, 0.3);
        border: 1px solid #313244;
        border-radius: 8px;
        color: #cbd5e0;
        font-size: 0.82rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .example-btn:hover {
        background: rgba(233, 69, 96, 0.15);
        border-color: #e94560;
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "calc_results" not in st.session_state:
    st.session_state.calc_results = None
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []

vectorstore = load_vectorstore()
kb_ready = vectorstore is not None

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>⚗️ ChemCopilot</h2>
        <p>AI Process Engineering Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    topic_filter = st.selectbox(
        "🔍 Topic Filter",
        options=[
            "All Topics",
            "Fluidization & Fluidized Beds",
            "Heat Exchangers",
            "Reactor Design (CSTR/PFR/PBR)",
            "Distillation",
            "Mass & Energy Balances",
            "Equipment Sizing"
        ],
        index=0,
        help="Filter the knowledge base retrieval to a specific topic area."
    )

    st.divider()

    with st.expander("🧮 Quick Calculations", expanded=False):
        calc_tab1, calc_tab2, calc_tab3 = st.tabs(["Fluidization", "Heat Exchanger", "Reactor"])

        with calc_tab1:
            st.markdown("##### Minimum Fluidization & Terminal Velocity")
            fl_dp = st.number_input("Particle diameter dp (µm)", value=300.0, min_value=1.0,
                                    step=10.0, key="fl_dp")
            fl_rho_p = st.number_input("Particle density ρ_p (kg/m³)", value=2500.0, min_value=100.0,
                                       step=50.0, key="fl_rho_p")
            fl_rho_g = st.number_input("Gas density ρ_g (kg/m³)", value=1.225, min_value=0.01,
                                       step=0.1, format="%.3f", key="fl_rho_g")
            fl_mu = st.number_input("Gas viscosity µ (Pa·s)", value=1.8e-5, min_value=1e-7,
                                    step=1e-6, format="%.2e", key="fl_mu")

            if st.button("Calculate Umf & Ut", key="btn_fluid", use_container_width=True):
                dp_m = fl_dp * 1e-6
                try:
                    umf_result = calc_umf_wen_yu(dp_m, fl_rho_p, fl_rho_g, fl_mu)
                    ut_result = calc_terminal_velocity(dp_m, fl_rho_p, fl_rho_g, fl_mu)
                    st.session_state.calc_results = {
                        "type": "fluidization",
                        "umf": umf_result,
                        "ut": ut_result
                    }

                    col_a, col_b = st.columns(2)
                    col_a.metric("U_mf", f"{umf_result['U_mf_ms']:.5f} m/s")
                    col_b.metric("U_t", f"{ut_result['U_t_ms']:.5f} m/s")
                    st.metric("Archimedes Number", f"{umf_result['Archimedes_number']:.2f}")
                    st.metric("Re_mf", f"{umf_result['Re_mf']:.4f}")

                    st.markdown(f"""<div class="result-highlight">
                        Regime: {umf_result['regime']}<br>
                        Drag regime: {ut_result['drag_regime']}<br>
                        Correlation: {umf_result['correlation']}
                    </div>""", unsafe_allow_html=True)

                    # Plotly bar chart for U_mf vs U_t
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=["U_mf (min. fluidization)", "U_t (terminal)"],
                        y=[umf_result['U_mf_ms'], ut_result['U_t_ms']],
                        marker_color=["#e94560", "#0f3460"],
                        text=[f"{umf_result['U_mf_ms']:.5f}", f"{ut_result['U_t_ms']:.5f}"],
                        textposition="outside"
                    ))
                    fig.update_layout(
                        title="Velocity Comparison",
                        yaxis_title="Velocity (m/s)",
                        template="plotly_dark",
                        height=300,
                        margin=dict(t=40, b=20, l=40, r=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Calculation error: {e}")

        with calc_tab2:
            st.markdown("##### LMTD & Area Sizing")
            hx_Thi = st.number_input("T_hot_in (°C)", value=180.0, key="hx_thi")
            hx_Tho = st.number_input("T_hot_out (°C)", value=120.0, key="hx_tho")
            hx_Tci = st.number_input("T_cold_in (°C)", value=30.0, key="hx_tci")
            hx_Tco = st.number_input("T_cold_out (°C)", value=70.0, key="hx_tco")
            hx_Q = st.number_input("Heat duty Q (kW)", value=500.0, min_value=0.1,
                                   step=10.0, key="hx_q")
            hx_U = st.number_input("U (W/m²·K)", value=350.0, min_value=1.0,
                                   step=10.0, key="hx_u")
            hx_flow = st.radio("Flow arrangement", ["counter", "parallel"],
                               index=0, key="hx_flow", horizontal=True)

            if st.button("Calculate LMTD & Area", key="btn_hx", use_container_width=True):
                try:
                    lmtd_res = calc_lmtd(hx_Thi, hx_Tho, hx_Tci, hx_Tco, flow=hx_flow)
                    if "error" in lmtd_res:
                        st.error(lmtd_res["error"])
                    else:
                        area_res = calc_heat_exchanger_area(
                            Q_W=hx_Q * 1000,
                            U_Wm2K=hx_U,
                            lmtd_K=lmtd_res["LMTD_K"]
                        )
                        st.session_state.calc_results = {
                            "type": "heat_exchanger",
                            "lmtd": lmtd_res,
                            "area": area_res
                        }

                        col_a, col_b = st.columns(2)
                        col_a.metric("LMTD", f"{lmtd_res['LMTD_K']:.3f} °C")
                        col_b.metric("Area", f"{area_res['area_m2']:.3f} m²")
                        st.metric("ΔT₁", f"{lmtd_res['delta_T1_K']:.1f} °C")
                        st.metric("ΔT₂", f"{lmtd_res['delta_T2_K']:.1f} °C")

                        st.markdown(f"""<div class="result-highlight">
                            Flow: {lmtd_res['flow_arrangement']}-current<br>
                            Eq: {area_res['reference']}
                        </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Calculation error: {e}")

        with calc_tab3:
            st.markdown("##### CSTR / PFR Volume")
            rx_type = st.radio("Reactor type", ["CSTR", "PFR"], index=0,
                               key="rx_type", horizontal=True)
            rx_FA0 = st.number_input("F_A0 (mol/s)", value=2.0, min_value=0.001,
                                     step=0.1, key="rx_fa0")
            rx_CA0 = st.number_input("C_A0 (mol/m³)", value=2.0, min_value=0.001,
                                     step=0.1, key="rx_ca0")
            rx_X = st.number_input("Conversion X", value=0.9, min_value=0.01,
                                   max_value=0.999, step=0.05, key="rx_x")
            rx_k = st.number_input("Rate constant k", value=0.05, min_value=1e-6,
                                   step=0.01, format="%.4f", key="rx_k")
            rx_n = st.number_input("Reaction order n", value=2.0, min_value=0.0,
                                   max_value=5.0, step=0.5, key="rx_n")
            rx_v0 = st.number_input("v₀ (m³/s)", value=1.0, min_value=0.001,
                                    step=0.1, key="rx_v0")

            if st.button("Calculate Volume", key="btn_rx", use_container_width=True):
                try:
                    if rx_type == "CSTR":
                        C_A_exit = rx_CA0 * (1 - rx_X)
                        neg_r_A = rx_k * C_A_exit**rx_n
                        if neg_r_A <= 0:
                            st.error("Reaction rate is zero at exit conversion — check inputs.")
                        else:
                            result = cstr_volume(rx_FA0, rx_X, neg_r_A)
                            st.session_state.calc_results = {"type": "reactor", "result": result}

                            st.metric("CSTR Volume", f"{result['volume_m3']:.4f} m³")
                            if result['space_time_s'] is not None:
                                st.metric("Space Time τ", f"{result['space_time_s']:.4f} s")
                            st.markdown(f"""<div class="result-highlight">
                                {result['reference']}
                            </div>""", unsafe_allow_html=True)
                    else:
                        result = pfr_volume_nth_order(rx_FA0, rx_CA0, rx_X, rx_k, rx_n, rx_v0)
                        st.session_state.calc_results = {"type": "reactor", "result": result}

                        st.metric("PFR Volume", f"{result['volume_m3']:.4f} m³")
                        st.metric("Damköhler #", f"{result['Damkohler_number']:.4f}")
                        st.markdown(f"""<div class="result-highlight">
                            {result['reference']}
                        </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Calculation error: {e}")

    st.divider()

    st.markdown("##### 💡 Example Queries")

    example_queries = [
        "Calculate Umf for 300µm sand in air at 400°C",
        "Size a counter-current HX: hot oil 180→120°C, cooling water 30→70°C, Q=500kW, U=350 W/m²K",
        "CSTR volume for 90% conversion, 2nd order, k=0.05 m³/mol·s, F_A0=2 mol/s, C_A0=2 mol/m³",
        "Bubbling bed regime: explain the two-phase theory of fluidization",
        "Compare CSTR vs PFR for exothermic reaction with cooling",
        "Estimate terminal velocity for 1mm glass beads in water"
    ]

    for i, eq in enumerate(example_queries):
        if st.button(eq, key=f"example_{i}", use_container_width=True):
            st.session_state.last_query = eq

    st.divider()

    st.markdown("##### 📦 Knowledge Base")
    if kb_ready:
        st.markdown("""<span class="status-badge status-ok">✅ Index loaded</span>""",
                    unsafe_allow_html=True)
    else:
        st.markdown("""<span class="status-badge status-warn">❌ No index found</span>""",
                    unsafe_allow_html=True)
        st.caption("Place PDFs in `sample_docs/` then run:\n```\npython -m ingest.build_index\n```")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True, key="btn_clear"):
        st.session_state.messages = []
        st.session_state.last_query = ""
        st.session_state.calc_results = None
        st.session_state.last_retrieved_docs = []
        st.rerun()


st.markdown("""
<div class="main-header">
    <h1>⚗️ ChemCopilot</h1>
    <p>Your AI-powered Chemical Process Engineering assistant — powered by RAG + NVIDIA NIM</p>
</div>
""", unsafe_allow_html=True)

if not kb_ready:
    st.warning(
        "⚠️ **Knowledge base not found.** The assistant will use general knowledge only.\n\n"
        "To enable RAG-powered answers:\n"
        "1. Place your ChemE PDF textbooks in the `sample_docs/` folder\n"
        "2. Run: `python -m ingest.build_index`\n"
        "3. Restart the app",
        icon="📚"
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.last_query:
    injected_query = st.session_state.last_query
    st.session_state.last_query = ""
else:
    injected_query = None

user_input = st.chat_input("Ask a chemical engineering question...")
query = injected_query or user_input

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔬 Retrieving from knowledge base & generating response..."):
            try:
                chat_history = []
                for m in st.session_state.messages[-7:-1]:
                    chat_history.append({"role": m["role"], "content": m["content"]})

                response_text, retrieved_docs = get_engineering_response(
                    user_query=query,
                    vectorstore=vectorstore,
                    topic_filter=topic_filter,
                    chat_history=chat_history if chat_history else None
                )

                st.markdown(response_text)
                st.session_state.last_retrieved_docs = retrieved_docs

            except Exception as e:
                response_text = f"⚠️ **Error:** {str(e)}\n\nPlease check your API key and network connection."
                st.error(response_text)
                retrieved_docs = []
                st.session_state.last_retrieved_docs = []

    st.session_state.messages.append({"role": "assistant", "content": response_text})

    if st.session_state.last_retrieved_docs:
        with st.expander("📚 Retrieved Sources", expanded=False):
            for i, doc in enumerate(st.session_state.last_retrieved_docs, 1):
                source = doc.metadata.get("source", "Unknown")
                topic = doc.metadata.get("topic", "general")
                chunk_id = doc.metadata.get("chunk_id", "?")
                st.markdown(
                    f"**{i}.** `{source}` — Topic: *{topic}* — Chunk #{chunk_id}"
                )
