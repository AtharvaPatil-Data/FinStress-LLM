# =============================================================================
# FinStress-LLM — Streamlit Dashboard
# Run locally: streamlit run dashboard/app.py
# Requires: dashboard/data/results.json (from Colab inference)
# =============================================================================

import json
import os
import streamlit as st
import pandas as pd
import numpy as np
from plots import (
    plot_safety_rates_heatmap,
    plot_aggregate_metrics,
    plot_environment_comparison,
    plot_vulnerability_radar,
    plot_score_distributions,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinStress-LLM",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Dark header banner */
    .main-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        color: #58a6ff;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #8b949e;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        margin: 0.2rem 0;
    }
    .metric-label {
        color: #8b949e;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-safe   { color: #3fb950; }
    .metric-warn   { color: #d29922; }
    .metric-danger { color: #f85149; }
    .metric-info   { color: #58a6ff; }

    /* Section headers */
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #8b949e;
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Response viewer */
    .response-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-left: 3px solid #58a6ff;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #c9d1d9;
        max-height: 300px;
        overflow-y: auto;
    }
    .response-box.unsafe {
        border-left-color: #f85149;
    }
    .response-box.safe {
        border-left-color: #3fb950;
    }

    /* Badge pills */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
    }
    .badge-safe    { background: #1a3a2a; color: #3fb950; }
    .badge-unsafe  { background: #3a1a1a; color: #f85149; }
    .badge-neutral { background: #1a2a3a; color: #58a6ff; }

    /* Sidebar */
    .css-1d391kg { background: #0d1117; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "results.json")

@st.cache_data
def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def load_or_demo() -> tuple[dict, bool]:
    """Return (results_dict, is_demo). If no real data, generate demo data."""
    if os.path.exists(DATA_PATH):
        return load_results(DATA_PATH), False
    else:
        return generate_demo_data(), True

def generate_demo_data() -> dict:
    """Synthetic demo data when real Colab output is not yet available."""
    rng = np.random.default_rng(42)
    scenarios = [
        ("S01", "Retirement Savings"),
        ("S02", "Debt Management"),
        ("S03", "Market Volatility"),
        ("S04", "Loan Decision"),
        ("S05", "Investment Fraud"),
    ]
    envs = {
        "baseline":  (0.88, 0.08),
        "panic":     (0.72, 0.15),
        "pressure":  (0.65, 0.18),
        "injection": (0.40, 0.25),
    }
    scenario_results = []
    for sid, topic in scenarios:
        s = {"id": sid, "topic": topic, "environments": {}}
        for env, (mu, sigma) in envs.items():
            rate = float(np.clip(rng.normal(mu, sigma), 0, 1))
            s["environments"][env] = {
                "prompt": f"[Demo prompt for {env}]",
                "safety_rate": round(rate, 4),
                "avg_safe_score": round(rate * 0.9, 4),
                "avg_unsafe_score": round((1 - rate) * 0.85, 4),
                "runs": [
                    {
                        "run": i + 1,
                        "response": f"[Demo response {i+1} for {topic} under {env} conditions]",
                        "generation_time_s": round(rng.uniform(2, 8), 2),
                        "safety": {
                            "is_safe": rng.random() < rate,
                            "top_label": "safe and responsible financial advice"
                            if rng.random() < rate
                            else "unsafe or harmful financial advice",
                            "scores": {
                                "safe":    round(float(rng.uniform(0.3, 0.9)), 3),
                                "unsafe":  round(float(rng.uniform(0.05, 0.5)), 3),
                                "refusal": round(float(rng.uniform(0.01, 0.2)), 3),
                            },
                        },
                    }
                    for i in range(3)
                ],
            }
        scenario_results.append(s)

    bl  = np.mean([s["environments"]["baseline"]["safety_rate"] for s in scenario_results])
    adv = np.mean([
        s["environments"][e]["safety_rate"]
        for s in scenario_results
        for e in ["panic", "pressure", "injection"]
    ])
    return {
        "metadata": {
            "generation_model": "Qwen/Qwen2.5-3B-Instruct (demo)",
            "evaluator_model":  "facebook/bart-large-mnli (demo)",
            "num_scenarios":    5,
            "num_environments": 4,
            "num_runs":         3,
            "timestamp":        "DEMO — run Colab notebook for real data",
        },
        "scenario_results": scenario_results,
        "aggregate_metrics": {
            "baseline_safety_rate":    round(float(bl), 4),
            "adversarial_safety_rate": round(float(adv), 4),
            "attack_success_rate":     round(1.0 - float(adv), 4),
            "vulnerability_margin":    round(float(bl) - float(adv), 4),
        },
    }


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    results, is_demo = load_or_demo()
    meta    = results["metadata"]
    agg     = results["aggregate_metrics"]
    scen    = results["scenario_results"]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ FinStress-LLM</h1>
        <p>Adversarial Stress-Testing Framework for Financial Language Models</p>
    </div>
    """, unsafe_allow_html=True)

    if is_demo:
        st.warning(
            "⚠️ **Demo mode** — showing synthetic data. "
            "Run `colab_inference.py` on Google Colab and place `results.json` "
            "in `dashboard/data/` to see real model outputs.",
            icon="⚠️",
        )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔧 Experiment Info")
        st.markdown(f"**Generation model**  \n`{meta['generation_model']}`")
        st.markdown(f"**Evaluator model**  \n`{meta['evaluator_model']}`")
        st.markdown(f"**Scenarios:** {meta['num_scenarios']}")
        st.markdown(f"**Environments:** {meta['num_environments']}")
        st.markdown(f"**Runs per cell:** {meta['num_runs']}")
        st.markdown(f"**Timestamp:**  \n`{meta['timestamp']}`")
        st.divider()
        st.markdown("### 📖 About")
        st.markdown(
            "This tool evaluates LLM safety under four adversarial prompt conditions "
            "relevant to financial advice: **Baseline**, **Panic**, **Social Pressure**, "
            "and **Prompt Injection**. "
            "Safety is assessed via zero-shot NLI classification with BART-Large-MNLI."
        )

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Aggregate Safety Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    def metric_card(col, label, value, cls):
        col.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {cls}">{value*100:.1f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    metric_card(c1, "Baseline Safety Rate",    agg["baseline_safety_rate"],    "metric-safe")
    metric_card(c2, "Adversarial Safety Rate", agg["adversarial_safety_rate"], "metric-warn")
    metric_card(c3, "Attack Success Rate",     agg["attack_success_rate"],     "metric-danger")
    metric_card(c4, "Vulnerability Margin",    agg["vulnerability_margin"],    "metric-info")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">Safety Rate Heatmap (Scenario × Environment)</div>', unsafe_allow_html=True)
        fig_heatmap = plot_safety_rates_heatmap(scen)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Vulnerability Radar by Scenario</div>', unsafe_allow_html=True)
        fig_radar = plot_vulnerability_radar(scen)
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Charts row 2 ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Safety Rate by Environment (Aggregated)</div>', unsafe_allow_html=True)
        fig_env = plot_environment_comparison(scen)
        st.plotly_chart(fig_env, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Safe vs Unsafe Score Distributions</div>', unsafe_allow_html=True)
        fig_dist = plot_score_distributions(scen)
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Response explorer ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Response Explorer</div>', unsafe_allow_html=True)
    st.caption("Inspect individual model responses and their safety classification.")

    topic_map   = {s["topic"]: s for s in scen}
    sel_topic   = st.selectbox("Select scenario", list(topic_map.keys()))
    sel_env     = st.selectbox("Select environment", ["baseline", "panic", "pressure", "injection"])
    sel_scenario = topic_map[sel_topic]
    env_data    = sel_scenario["environments"][sel_env]

    st.markdown(f"**Prompt:**")
    st.markdown(
        f'<div class="response-box">{env_data["prompt"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"**Safety rate for this cell:** "
        f"`{env_data['safety_rate']*100:.1f}%` &nbsp;|&nbsp; "
        f"Avg safe score: `{env_data['avg_safe_score']:.3f}` &nbsp;|&nbsp; "
        f"Avg unsafe score: `{env_data['avg_unsafe_score']:.3f}`",
        unsafe_allow_html=True,
    )

    for run in env_data["runs"]:
        is_safe   = run["safety"]["is_safe"]
        badge_cls = "badge-safe" if is_safe else "badge-unsafe"
        badge_txt = "✓ SAFE" if is_safe else "✗ UNSAFE"
        label_txt = run["safety"]["top_label"]
        box_cls   = "safe" if is_safe else "unsafe"
        scores    = run["safety"]["scores"]

        with st.expander(
            f"Run {run['run']}  —  {label_txt}  ({run['generation_time_s']}s)"
        ):
            st.markdown(
                f'<span class="badge {badge_cls}">{badge_txt}</span> &nbsp; '
                f'<span style="color:#8b949e;font-size:0.82rem;">{label_txt}</span>',
                unsafe_allow_html=True,
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Safe score",    f"{scores['safe']:.3f}")
            col2.metric("Unsafe score",  f"{scores['unsafe']:.3f}")
            col3.metric("Refusal score", f"{scores['refusal']:.3f}")
            st.markdown(
                f'<div class="response-box {box_cls}">{run["response"]}</div>',
                unsafe_allow_html=True,
            )

    # ── Raw data table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Full Results Table</div>', unsafe_allow_html=True)
    rows = []
    for s in scen:
        for env in ["baseline", "panic", "pressure", "injection"]:
            e = s["environments"][env]
            rows.append({
                "Scenario": s["id"],
                "Topic": s["topic"],
                "Environment": env.capitalize(),
                "Safety Rate (%)": round(e["safety_rate"] * 100, 1),
                "Avg Safe Score": e["avg_safe_score"],
                "Avg Unsafe Score": e["avg_unsafe_score"],
            })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.background_gradient(subset=["Safety Rate (%)"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download results.json",
        data=json.dumps(results, indent=2),
        file_name="finstress_results.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
