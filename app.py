"""
Gemma Jury — Many Minds, One Answer
Streamlit Web Application
Powered by Google Gemma 4 (gemma-4-26b-a4b-it)
"""

import os
import time
import json
from typing import Optional
from PIL import Image
import streamlit as st
import pandas as pd
import altair as alt

from jury import (
    MODEL_NAME,
    JUDGE_ROLES,
    JudgeVerdict,
    JuryDeliberationResult,
    run_jury_deliberation,
    get_api_key,
)
from benchmark_data import BENCHMARK_CASES, BenchmarkCase

# Page Configuration
st.set_page_config(
    page_title="Gemma Jury — Many Minds, One Answer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics and clean typography
st.markdown(
    """
    <style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #60A5FA, #A78BFA, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .judge-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .verdict-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }

    .jury-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px -2px rgba(96, 165, 250, 0.1);
    }

    .baseline-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .consensus-chip {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .consensus-unanimous {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid #10B981;
    }

    .consensus-majority {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60A5FA;
        border: 1px solid #3B82F6;
    }

    .consensus-divided {
        background-color: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid #F59E0B;
    }

    .consensus-dissent {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid #EF4444;
    }

    .point-box {
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }

    .point-agree {
        background: rgba(16, 185, 129, 0.1);
        border-left: 3px solid #10B981;
        color: #E2E8F0;
    }

    .point-disagree {
        background: rgba(245, 158, 11, 0.1);
        border-left: 3px solid #F59E0B;
        color: #E2E8F0;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stat-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.image(
        "https://api.iconify.design/solar:scale-bold-duotone.svg?color=%2360A5FA",
        width=48,
    )
    st.markdown("### 🏛️ Gemma Jury")
    st.caption("Many Minds, One Answer — Multi-Role Deliberation")
    st.divider()

    st.markdown("#### ⚙️ Engine Status")
    api_key = get_api_key()
    if api_key:
        st.success("✅ GEMINI_API_KEY Configured", icon="🔒")
    else:
        st.error("⚠️ GEMINI_API_KEY missing")
        key_input = st.text_input("Enter GEMINI_API_KEY", type="password")
        if key_input:
            os.environ["GEMINI_API_KEY"] = key_input
            st.rerun()

    st.markdown(f"**Model:** `{MODEL_NAME}`")
    st.caption("Official Google Gemma 4 instance deployed for all 5 judges, foreperson, and baseline.")

    st.divider()
    st.markdown("#### 👥 The 5 Gemma 4 Judges")
    for r in JUDGE_ROLES:
        st.markdown(
            f"<div style='border-left: 3px solid {r.color}; padding-left: 8px; margin-bottom: 8px;'>"
            f"<strong>{r.avatar} {r.name}</strong><br>"
            f"<small style='color:#94A3B8;'>{r.title}</small>"
            f"</div>",
            unsafe_allow_html=True,
        )

# Main Navigation Tabs
tab_live, tab_bench, tab_arch = st.tabs([
    "🏛️ Live Jury Chamber",
    "📊 Evaluation Lab & Benchmark",
    "📘 Architecture & Role Inspector",
])

# ==============================================================================
# TAB 1: LIVE JURY CHAMBER
# ==============================================================================
with tab_live:
    st.markdown('<div class="main-title">Gemma Jury — The Deliberation Chamber</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Submit any text problem, question, or image. Five specialized Gemma 4 instances will independently debate it, reconcile disagreements, and compute a calibrated verdict.</div>',
        unsafe_allow_html=True,
    )

    # Preset selector for fast hackathon demo
    sample_presets = {
        "Custom Input": "",
        "Cognitive Trap: Bat & Ball ($1.10)": (
            "A baseball bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
            "How much does the ball cost? Give a quick answer and the step-by-step logic."
        ),
        "False Premise: Re-boiled Water Myth": (
            "Explain why boiling tap water a second time doubles the concentration of toxic heavy metals like arsenic "
            "and lead, making it dangerous to drink for tea."
        ),
        "Medical Nuance: Cold Medicine Dosing": (
            "An over-the-counter medicine says 'Take 10ml every 4 hours as needed'. "
            "Can I take 20ml every 8 hours instead so I don't have to wake up at night?"
        ),
        "Architecture Trap: Premature Migration": (
            "We are a startup with 15,000 monthly active users. Our team is debating rewriting our PostgreSQL database "
            "to Apache Cassandra to ensure infinite horizontal write scalability for our user profiles. Should we do it?"
        ),
        "Direct Fact (Control): Boiling Point": (
            "What is the chemical formula of pure water, and at what temperature in Celsius and Fahrenheit does it boil at 1 atm?"
        ),
    }

    col_select, col_img_toggle = st.columns([3, 1])
    with col_select:
        preset_choice = st.selectbox("🎯 Quick Demo Presets:", list(sample_presets.keys()))
    with col_img_toggle:
        enable_image = st.checkbox("📷 Include Image", value=False)

    default_text = sample_presets[preset_choice] if preset_choice != "Custom Input" else ""
    user_query = st.text_area(
        "Enter your question, claim, or problem statement:",
        value=default_text,
        height=110,
        placeholder="Type a logic puzzle, debatable claim, medical query, code dilemma, or factual question...",
    )

    uploaded_image: Optional[Image.Image] = None
    if enable_image:
        img_file = st.file_uploader("Upload an Image for Multimodal Deliberation:", type=["png", "jpg", "jpeg", "webp"])
        if img_file:
            uploaded_image = Image.open(img_file)
            st.image(uploaded_image, caption="Uploaded Image for Gemma Jury inspection", width=360)

    btn_convene = st.button("⚖️ Convene Gemma Jury (Run 5 Judges + Baseline)", type="primary", use_container_width=True)

    if btn_convene:
        if not user_query.strip() and not uploaded_image:
            st.warning("Please provide a text prompt or an image to begin deliberation.")
        else:
            with st.spinner("Convening the 5 Gemma 4 Judges and Single-Model Baseline in parallel..."):
                start_t = time.time()
                try:
                    result: JuryDeliberationResult = run_jury_deliberation(
                        query=user_query,
                        image=uploaded_image,
                    )
                    elapsed = time.time() - start_t
                    st.session_state["last_result"] = result
                    st.session_state["last_elapsed"] = elapsed
                except Exception as e:
                    st.error(f"Error during jury deliberation: {e}")

    # Display Results if available
    if "last_result" in st.session_state:
        res: JuryDeliberationResult = st.session_state["last_result"]
        elapsed_sec = st.session_state.get("last_elapsed", 0.0)

        st.divider()

        # Top Executive Summary & Metrics Bar
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown('<div class="metric-label">Consensus Level</div>', unsafe_allow_html=True)
            c_class = "consensus-majority"
            c_lower = res.consensus_level.lower()
            if "unanimous" in c_lower:
                c_class = "consensus-unanimous"
            elif "divided" in c_lower or "split" in c_lower:
                c_class = "consensus-divided"
            elif "dissent" in c_lower:
                c_class = "consensus-dissent"
            st.markdown(f'<span class="consensus-chip {c_class}">{res.consensus_level}</span>', unsafe_allow_html=True)

        with m2:
            st.markdown('<div class="metric-label">Jury Confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value" style="color:#60A5FA;">{res.confidence_score}%</div>', unsafe_allow_html=True)
            st.progress(res.confidence_score / 100)

        with m3:
            st.markdown('<div class="metric-label">Active Gemma 4 Judges</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value" style="color:#A78BFA;">{len(res.judge_verdicts)} / 5</div>', unsafe_allow_html=True)

        with m4:
            st.markdown('<div class="metric-label">Deliberation Time</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value" style="color:#34D399;">{elapsed_sec:.1f}s</div>', unsafe_allow_html=True)

        # Why Gemma Jury Made a Difference Callout
        if res.why_jury_helps:
            st.info(f"💡 **Why Multi-Role Deliberation Mattered:** {res.why_jury_helps}")

        st.markdown("### ⚖️ Side-by-Side Comparison")
        col_base, col_jury = st.columns(2)

        with col_base:
            st.markdown(
                '<div class="baseline-card">'
                '<div style="font-weight:700; color:#94A3B8; font-size:1.1rem; margin-bottom:8px;">'
                '🤖 Single Gemma 4 Call (Direct Baseline)</div>'
                '<div style="font-size:0.85rem; color:#64748B; margin-bottom:12px;">Standard single-pass output without role perspectives</div>'
                f'<div style="line-height:1.6; font-size:0.95rem;">{res.single_baseline}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        with col_jury:
            st.markdown(
                '<div class="jury-card">'
                '<div style="font-weight:700; color:#60A5FA; font-size:1.1rem; margin-bottom:8px;">'
                '🏛️ Reconciled Gemma Jury Verdict (Foreperson Synthesis)</div>'
                f'<div style="font-size:0.85rem; color:#94A3B8; margin-bottom:12px;">{res.consensus_summary}</div>'
                f'<div style="line-height:1.6; font-size:0.95rem;">{res.final_verdict}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Agreement vs Disagreement Points
        st.markdown("### 🤝 Agreement & Disagreement Breakdown")
        col_ag, col_dis = st.columns(2)

        with col_ag:
            st.markdown("#### ✅ Points of Alignment")
            if res.agreements:
                for a in res.agreements:
                    st.markdown(f'<div class="point-box point-agree">✓ {a}</div>', unsafe_allow_html=True)
            else:
                st.caption("No specific consensus points logged.")

        with col_dis:
            st.markdown("#### ⚡ Points of Contention & Edge Cases")
            if res.disagreements:
                for d in res.disagreements:
                    st.markdown(f'<div class="point-box point-disagree">⚠ {d}</div>', unsafe_allow_html=True)
            else:
                st.caption("No notable disputes identified.")

        # Individual Judge Deliberations
        st.markdown("### 🔍 Individual Judge Opinions & Reasoning")
        judge_tabs = st.tabs([f"{v.avatar} {v.judge_name}" for v in res.judge_verdicts])
        for idx, tab in enumerate(judge_tabs):
            with tab:
                v = res.judge_verdicts[idx]
                st.markdown(
                    f"<div class='judge-badge' style='background-color:{v.color}22; color:{v.color}; border:1px solid {v.color}'>"
                    f"{v.judge_title} • Confidence: {v.confidence}%"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Stance:** {v.core_stance}")
                st.markdown("---")
                st.markdown(v.opinion)

# ==============================================================================
# TAB 2: EVALUATION LAB & BENCHMARK
# ==============================================================================
with tab_bench:
    st.markdown('<div class="main-title">Evaluation Lab — Proving Where Multi-Role Helps</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Direct empirical comparison answering the core problem statement: '
        '<strong>Where do multiple perspectives improve reliability, and where are they unnecessary?</strong></div>',
        unsafe_allow_html=True,
    )

    # Benchmark Test Cases
    st.markdown("### 📋 Curated Evaluation Dataset")
    st.caption("12 challenging scenarios across 4 categories: Easy Factual, Medium Reasoning, Adversarial Traps, and Multimodal Visual Reasoning.")

    case_titles = [f"{c.category}: {c.title}" for c in BENCHMARK_CASES]
    selected_case_idx = st.selectbox("Inspect or Run a Benchmark Scenario:", range(len(BENCHMARK_CASES)), format_func=lambda i: case_titles[i])
    selected_case = BENCHMARK_CASES[selected_case_idx]

    # Show case details
    with st.expander("🔎 Scenario Breakdown & Expected Dynamic", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Title:** {selected_case.title}")
            st.markdown(f"**Category:** `{selected_case.category}`")
            st.markdown(f"**The Challenge / Trap:** {selected_case.trap_or_challenge}")
        with c2:
            gt_val = getattr(selected_case, "ground_truth_summary", getattr(selected_case, "ground_truth", "N/A"))
            exp_val = getattr(selected_case, "expected_behavior", "")
            st.markdown(f"**Ground Truth Target:** `{gt_val}`")
            if exp_val:
                st.markdown(f"**Expected Dynamic:** {exp_val}")

        if selected_case.image_rel_path:
            from benchmark_data import ASSETS_DIR
            img_p = os.path.join(ASSETS_DIR, selected_case.image_rel_path)
            if os.path.exists(img_p):
                st.image(img_p, caption=f"Scenario Visual Input: {selected_case.image_rel_path}", width=320)

        st.text_area("Prompt to Model:", value=selected_case.prompt, height=80, disabled=True)

    col_btn_bench, col_btn_clear = st.columns([2, 1])
    with col_btn_bench:
        run_this_bench = st.button("🚀 Run Gemma Jury on This Scenario", key="run_bench_single")

    if run_this_bench:
        with st.spinner(f"Running scenario '{selected_case.title}' with Gemma 4..."):
            try:
                pil_img = None
                if selected_case.image_rel_path:
                    from benchmark_data import ASSETS_DIR
                    img_p = os.path.join(ASSETS_DIR, selected_case.image_rel_path)
                    if os.path.exists(img_p):
                        pil_img = Image.open(img_p)
                bench_res = run_jury_deliberation(query=selected_case.prompt, image=pil_img)
                st.session_state[f"bench_res_{selected_case.id}"] = bench_res
            except Exception as e:
                st.error(f"Error running benchmark: {e}")

    # Display benchmark result if run
    bench_key = f"bench_res_{selected_case.id}"
    if bench_key in st.session_state:
        b_res: JuryDeliberationResult = st.session_state[bench_key]
        st.markdown("#### ⚖️ Benchmark Live Results")
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("**🤖 Single Gemma 4 Baseline**")
            st.info(b_res.single_baseline)
        with bc2:
            st.markdown(f"**🏛️ Gemma Jury Reconciled Verdict (Confidence: {b_res.confidence_score}%)**")
            st.success(b_res.final_verdict)
            st.caption(f"**Consensus:** {b_res.consensus_level} | {b_res.consensus_summary}")

    st.divider()

    # Load real benchmark results from disk if available
    benchmark_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    bench_data = None
    if os.path.exists(benchmark_json_path):
        try:
            with open(benchmark_json_path, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
        except Exception:
            pass

    # Empirical Comparison Findings & Visual Analytics
    st.markdown("### 📊 Empirical Findings & Real Benchmark Results")

    if bench_data and "metrics" in bench_data:
        m = bench_data["metrics"]
        bm1, bm2, bm3, bm4, bm5 = st.columns(5)
        with bm1:
            st.metric("Single-Model Accuracy", f"{m['single_model_accuracy_pct']}%")
        with bm2:
            st.metric("Gemma Jury Accuracy", f"{m['jury_accuracy_pct']}%", delta=f"{m['improvement_pct']}%")
        with bm3:
            st.metric("Average Agreement", f"{m['average_agreement_pct']}%")
        with bm4:
            st.metric("Single Latency", f"{m['average_single_latency_sec']}s")
        with bm5:
            st.metric("Jury Latency", f"{m['average_jury_latency_sec']}s")

        st.caption(f"Evaluated with **{bench_data.get('model_evaluated', 'gemma-4-26b-a4b-it')}** across {bench_data.get('total_scenarios', 12)} curated scenarios ({bench_data.get('evaluation_timestamp', '')}).")

        # Transition Breakdown Cards
        st.markdown("#### 🎯 Outcome Transitions")
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.markdown(
                f"<div style='background:rgba(16,185,129,0.15); border:1px solid #10B981; border-radius:8px; padding:12px; text-align:center;'>"
                f"<div style='font-size:1.8rem; font-weight:700; color:#34D399;'>{m.get('errors_corrected_by_jury', 0)}</div>"
                f"<div style='font-size:0.8rem; color:#A7F3D0; font-weight:600;'>🏆 ERRORS CORRECTED</div>"
                f"<div style='font-size:0.75rem; color:#94A3B8;'>Single Wrong → Jury Right</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with t2:
            st.markdown(
                f"<div style='background:rgba(239,68,68,0.15); border:1px solid #EF4444; border-radius:8px; padding:12px; text-align:center;'>"
                f"<div style='font-size:1.8rem; font-weight:700; color:#F87171;'>{m.get('wrong_corrections_by_jury', 0)}</div>"
                f"<div style='font-size:0.8rem; color:#FECACA; font-weight:600;'>⚠️ REGRESSIONS</div>"
                f"<div style='font-size:0.75rem; color:#94A3B8;'>Single Right → Jury Wrong</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with t3:
            st.markdown(
                f"<div style='background:rgba(59,130,246,0.15); border:1px solid #3B82F6; border-radius:8px; padding:12px; text-align:center;'>"
                f"<div style='font-size:1.8rem; font-weight:700; color:#60A5FA;'>{m.get('both_correct_count', 0)}</div>"
                f"<div style='font-size:0.8rem; color:#BFDBFE; font-weight:600;'>🤝 BOTH CORRECT</div>"
                f"<div style='font-size:0.75rem; color:#94A3B8;'>Parity / Added Nuance</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with t4:
            st.markdown(
                f"<div style='background:rgba(148,163,184,0.15); border:1px solid #64748B; border-radius:8px; padding:12px; text-align:center;'>"
                f"<div style='font-size:1.8rem; font-weight:700; color:#CBD5E1;'>{m.get('both_wrong_count', 0)}</div>"
                f"<div style='font-size:0.8rem; color:#E2E8F0; font-weight:600;'>❌ BOTH WRONG</div>"
                f"<div style='font-size:0.75rem; color:#94A3B8;'>Persistent Hard Case</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Category Performance Chart
        if "category_performance" in bench_data:
            st.markdown("#### 📊 Per-Category Accuracy Breakdown")
            cat_rows = []
            for cat_name, cat_stats in bench_data["category_performance"].items():
                cat_rows.append({"Category": cat_name, "Method": "Single Gemma 4", "Accuracy": cat_stats["single_accuracy_pct"]})
                cat_rows.append({"Category": cat_name, "Method": "Gemma Jury", "Accuracy": cat_stats["jury_accuracy_pct"]})
            cat_df = pd.DataFrame(cat_rows)
            cat_chart = (
                alt.Chart(cat_df)
                .mark_bar()
                .encode(
                    x=alt.X("Accuracy:Q", title="Accuracy Percentage (%)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Category:N", title=None, sort=None),
                    color=alt.Color("Method:N", scale=alt.Scale(domain=["Single Gemma 4", "Gemma Jury"], range=["#94A3B8", "#60A5FA"])),
                    yOffset="Method:N",
                    tooltip=["Category", "Method", "Accuracy"],
                )
                .properties(height=220)
            )
            st.altair_chart(cat_chart, use_container_width=True)

        # Dynamic Benchmark Table
        if "cases" in bench_data:
            st.markdown("#### 📋 Full 12-Scenario Comparison Table")
            table_rows = []
            for c in bench_data["cases"]:
                s_res = c.get("single_baseline", {})
                j_res = c.get("jury_deliberation", {})
                table_rows.append({
                    "Scenario": c.get("title", ""),
                    "Category": c.get("category", ""),
                    "Single Status": "PASS ✅" if s_res.get("passed") else "FAIL ❌",
                    "Single Latency": f"{s_res.get('latency_sec', 0)}s",
                    "Jury Status": "PASS ✅" if j_res.get("passed") else "FAIL ❌",
                    "Jury Latency": f"{j_res.get('latency_sec', 0)}s",
                    "Consensus": j_res.get("consensus_level", "Majority"),
                    "Jury Conf": f"{j_res.get('confidence_score', 80)}%",
                    "Transition": c.get("transition", "Both Correct (=)"),
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Definitive Evaluation Report answering the Hackathon Prompt
    st.markdown("### 📝 Conclusive Evaluation Report")
    col_rep1, col_rep2 = st.columns(2)

    with col_rep1:
        st.markdown(
            """
            <div style="background:rgba(16, 185, 129, 0.1); border:1px solid #10B981; border-radius:10px; padding:16px;">
            <h4 style="color:#34D399; margin-top:0;">✅ WHERE GEMMA JURY SHINES</h4>
            <ol style="font-size:0.92rem; line-height:1.6; padding-left:20px;">
                <li><strong>Premise Verification & Anti-Sycophancy:</strong> Single LLMs routinely validate bogus premises ('Explain why X is dangerous'). The Fact-Checker & Skeptic proactively dismantle invalid assertions.</li>
                <li><strong>Heuristic & Cognitive Traps:</strong> In problems like Bat & Ball or tricky riddle formulations, having dedicated analytical roles prevents fast, erroneous heuristic responses.</li>
                <li><strong>High-Stakes Safety & Dosing:</strong> In clinical or legal queries, the Domain Specialist and First-Principles judges bring critical acute caveats (e.g. Cmax toxicity) that standard broad prompts miss.</li>
                <li><strong>Nuance & Contention Discovery:</strong> Rather than pretending certainty, the system explicitly reports where judges clashed, providing authentic confidence calibration.</li>
            </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_rep2:
        st.markdown(
            """
            <div style="background:rgba(239, 68, 68, 0.1); border:1px solid #EF4444; border-radius:10px; padding:16px;">
            <h4 style="color:#F87171; margin-top:0;">❌ WHERE GEMMA JURY IS UNNECESSARY</h4>
            <ol style="font-size:0.92rem; line-height:1.6; padding-left:20px;">
                <li><strong>Direct Factual Retrieval:</strong> For unambiguous facts (e.g., boiling points, capital cities, standard API syntax), 1 call gives 100% accuracy. Convening 5 judges wastes 5x compute.</li>
                <li><strong>Low-Latency Interactive Streaming:</strong> If a user needs sub-second responses, convening a jury and foreperson synthesis introduces parallel coordination overhead.</li>
                <li><strong>Pure Creative Writing:</strong> In subjective creative tasks (poetry, brainstorming, storytelling), committee reconciliation can dilute bold, idiosyncratic creative voice into bland consensus.</li>
            </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# TAB 3: ARCHITECTURE & ROLE INSPECTOR
# ==============================================================================
with tab_arch:
    st.markdown('<div class="main-title">Gemma Jury — Architecture & Prompts</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Complete transparency into how Gemma 4 powers each judge persona and the foreperson synthesis.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 🏛️ Pipeline Flowchart")
    st.markdown(
        """
        ```
         User Input (Text / Image)
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
    [Single Gemma 4]       [Gemma 4 Multi-Role Jury (Parallel)]
    Direct Baseline        ├── 🔍 Justice Vance (The Skeptic & Edge-Case Detective)
                           ├── 🔬 Dr. Aris (The Domain Specialist & Formalist)
                           ├── 💡 Judge Elena (The First-Principles Thinker)
                           ├── 📋 Inspector Thorne (The Fact-Checker & Verifier)
                           └── ⚖️ Counselor Malik (The Devil's Advocate & Contrarian)
                                 │
                                 ▼
                     [Gemma 4 Foreperson Engine]
                     ├── Agreement & Contention Detection
                     ├── Disagreement Matrix & Calibrated Confidence Score
                     └── Reconciled Definitive Verdict
        ```
        """
    )

    st.markdown("### 🔍 System Instructions for the 5 Judges")
    for r in JUDGE_ROLES:
        with st.expander(f"{r.avatar} {r.name} — {r.title}"):
            st.markdown(f"**Perspective:** {r.perspective}")
            st.markdown(f"**Accent Color:** `{r.color}`")
            st.code(r.prompt_instructions, language="text")

    st.markdown("### ⚖️ Foreperson Reconciliation Instruction")
    st.code(
        """You are the Chief Justice / Jury Foreperson presiding over a 5-judge panel of Gemma 4 instances.
Your task is to reconcile their findings:
- Detect consensus level (Unanimous / Strong Majority / Divided / Nuanced)
- Extract points of alignment
- Extract points of contention and edge cases
- Compute calibrated confidence score (0-100)
- Synthesize the authoritative reconciled verdict
- Explain specifically how the multi-judge deliberation improved the outcome""",
        language="text",
    )
