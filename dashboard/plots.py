# =============================================================================
# FinStress-LLM — Plotly Chart Functions
# =============================================================================

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Theme constants ───────────────────────────────────────────────────────────
BG        = "#0d1117"
PAPER_BG  = "#161b22"
GRID      = "#21262d"
TEXT      = "#c9d1d9"
MUTED     = "#8b949e"
SAFE      = "#3fb950"
WARN      = "#d29922"
DANGER    = "#f85149"
BLUE      = "#58a6ff"
PURPLE    = "#bc8cff"
ENV_COLORS = {
    "baseline":  BLUE,
    "panic":     WARN,
    "pressure":  PURPLE,
    "injection": DANGER,
}
ENV_ORDER = ["baseline", "panic", "pressure", "injection"]

LAYOUT_DEFAULTS = dict(
    plot_bgcolor=BG,
    paper_bgcolor=PAPER_BG,
    font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=11),
    margin=dict(l=40, r=20, t=40, b=40),
)


def _base_layout(**kwargs):
    layout = dict(**LAYOUT_DEFAULTS)
    layout.update(kwargs)
    return layout


# ── 1. Heatmap ────────────────────────────────────────────────────────────────

def plot_safety_rates_heatmap(scenario_results: list) -> go.Figure:
    """Safety rate heatmap: scenarios (rows) × environments (columns)."""
    topics = [s["topic"] for s in scenario_results]
    envs   = ENV_ORDER

    z = []
    for s in scenario_results:
        row = [s["environments"][env]["safety_rate"] * 100 for env in envs]
        z.append(row)

    z_text = [[f"{v:.1f}%" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[e.capitalize() for e in envs],
            y=topics,
            text=z_text,
            texttemplate="%{text}",
            textfont=dict(size=12, family="IBM Plex Mono, monospace"),
            colorscale=[
                [0.0,  "#3a1a1a"],
                [0.4,  "#5c2a1a"],
                [0.6,  "#5c4a1a"],
                [0.8,  "#1a3a2a"],
                [1.0,  "#3fb950"],
            ],
            zmin=0, zmax=100,
            showscale=True,
            colorbar=dict(
                title=dict(text="Safety %", font=dict(color=MUTED, size=10)),
                tickfont=dict(color=MUTED, size=9),
                thickness=12,
                len=0.8,
            ),
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text="Safety Rate Heatmap", font=dict(size=13, color=MUTED)),
            xaxis=dict(
                tickfont=dict(color=TEXT),
                showgrid=False,
                side="top",
            ),
            yaxis=dict(
                tickfont=dict(color=TEXT),
                showgrid=False,
                autorange="reversed",
            ),
        )
    )
    return fig


# ── 2. Radar chart ────────────────────────────────────────────────────────────

def plot_vulnerability_radar(scenario_results: list) -> go.Figure:
    """Radar showing vulnerability margin per scenario."""
    topics = [s["topic"] for s in scenario_results]

    # Vulnerability margin = baseline_rate - avg_adversarial_rate
    margins = []
    for s in scenario_results:
        bl  = s["environments"]["baseline"]["safety_rate"]
        adv = np.mean([
            s["environments"][e]["safety_rate"]
            for e in ["panic", "pressure", "injection"]
        ])
        margins.append(round((bl - adv) * 100, 1))

    # Close the radar
    topics_closed  = topics  + [topics[0]]
    margins_closed = margins + [margins[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=margins_closed,
        theta=topics_closed,
        fill="toself",
        fillcolor=f"rgba(248, 81, 73, 0.15)",
        line=dict(color=DANGER, width=2),
        marker=dict(size=6, color=DANGER),
        name="Vulnerability Margin (%)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(
                visible=True,
                range=[0, max(margins) * 1.3],
                tickfont=dict(color=MUTED, size=9),
                gridcolor=GRID,
                linecolor=GRID,
            ),
            angularaxis=dict(
                tickfont=dict(color=TEXT, size=10),
                gridcolor=GRID,
                linecolor=GRID,
            ),
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=BG,
        font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=11),
        margin=dict(l=40, r=40, t=50, b=40),
        title=dict(text="Vulnerability Margin by Scenario", font=dict(size=13, color=MUTED)),
        showlegend=False,
    )
    return fig


# ── 3. Environment comparison bar ────────────────────────────────────────────

def plot_environment_comparison(scenario_results: list) -> go.Figure:
    """Grouped bar chart: average safety rate per environment across all scenarios."""
    env_avg = {}
    env_std = {}
    for env in ENV_ORDER:
        rates = [s["environments"][env]["safety_rate"] * 100 for s in scenario_results]
        env_avg[env] = np.mean(rates)
        env_std[env] = np.std(rates)

    fig = go.Figure()
    for env in ENV_ORDER:
        fig.add_trace(go.Bar(
            x=[env.capitalize()],
            y=[env_avg[env]],
            error_y=dict(
                type="data",
                array=[env_std[env]],
                visible=True,
                color=MUTED,
                thickness=1.5,
            ),
            name=env.capitalize(),
            marker=dict(
                color=ENV_COLORS[env],
                opacity=0.85,
                line=dict(color=ENV_COLORS[env], width=1),
            ),
            text=[f"{env_avg[env]:.1f}%"],
            textposition="outside",
            textfont=dict(size=12, color=ENV_COLORS[env]),
        ))

    fig.update_layout(
        **_base_layout(
            title=dict(text="Mean Safety Rate by Environment", font=dict(size=13, color=MUTED)),
            xaxis=dict(showgrid=False, tickfont=dict(color=TEXT)),
            yaxis=dict(
                range=[0, 115],
                showgrid=True,
                gridcolor=GRID,
                ticksuffix="%",
                tickfont=dict(color=MUTED),
                title=dict(text="Safety Rate", font=dict(color=MUTED, size=10)),
            ),
            showlegend=False,
            bargap=0.35,
        )
    )
    return fig


# ── 4. Score distributions violin ────────────────────────────────────────────

def plot_score_distributions(scenario_results: list) -> go.Figure:
    """Violin plot of safe/unsafe NLI scores across all runs and environments."""
    safe_scores   = []
    unsafe_scores = []

    for s in scenario_results:
        for env in ENV_ORDER:
            for run in s["environments"][env]["runs"]:
                safe_scores.append(run["safety"]["scores"]["safe"])
                unsafe_scores.append(run["safety"]["scores"]["unsafe"])

    fig = go.Figure()
    fig.add_trace(go.Violin(
        y=safe_scores,
        name="Safe score",
        side="negative",
        line_color=SAFE,
        fillcolor=f"rgba(63, 185, 80, 0.2)",
        meanline_visible=True,
        points="outliers",
        pointpos=-0.5,
        marker=dict(color=SAFE, size=3, opacity=0.5),
    ))
    fig.add_trace(go.Violin(
        y=unsafe_scores,
        name="Unsafe score",
        side="positive",
        line_color=DANGER,
        fillcolor=f"rgba(248, 81, 73, 0.2)",
        meanline_visible=True,
        points="outliers",
        pointpos=0.5,
        marker=dict(color=DANGER, size=3, opacity=0.5),
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text="NLI Score Distributions (All Runs)", font=dict(size=13, color=MUTED)),
            violinmode="overlay",
            yaxis=dict(
                showgrid=True,
                gridcolor=GRID,
                tickfont=dict(color=MUTED),
                title=dict(text="NLI Score", font=dict(color=MUTED, size=10)),
                range=[0, 1],
            ),
            xaxis=dict(showticklabels=False, showgrid=False),
            legend=dict(
                font=dict(color=TEXT, size=10),
                bgcolor=PAPER_BG,
                bordercolor=GRID,
                borderwidth=1,
            ),
        )
    )
    return fig


# ── 5. Aggregate metrics bar (unused in app but exported for reports) ─────────

def plot_aggregate_metrics(agg: dict) -> go.Figure:
    """Horizontal bar chart of all four aggregate metrics."""
    labels = [
        "Baseline Safety Rate",
        "Adversarial Safety Rate",
        "Attack Success Rate",
        "Vulnerability Margin",
    ]
    values = [
        agg["baseline_safety_rate"] * 100,
        agg["adversarial_safety_rate"] * 100,
        agg["attack_success_rate"] * 100,
        agg["vulnerability_margin"] * 100,
    ]
    colors = [SAFE, WARN, DANGER, BLUE]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, opacity=0.85),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=12),
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text="Aggregate Metrics", font=dict(size=13, color=MUTED)),
            xaxis=dict(
                range=[0, 120],
                ticksuffix="%",
                showgrid=True,
                gridcolor=GRID,
                tickfont=dict(color=MUTED),
            ),
            yaxis=dict(showgrid=False, tickfont=dict(color=TEXT)),
        )
    )
    return fig


# ── 6. Cognitive Bias Heatmap ────────────────────────────────────────────────

def plot_cognitive_bias_heatmap(scenario_results: list) -> go.Figure:
    """Heatmap showing distribution of cognitive biases across scenarios and environments."""
    topics = [s["topic"] for s in scenario_results]
    envs = ENV_ORDER
    
    # Collect bias scores for each cell
    bias_types = ["herd_mentality", "loss_aversion", "overconfidence", "neutral"]
    
    # For each scenario-environment pair, get the dominant bias
    z = []
    hover_text = []
    
    for s in scenario_results:
        row = []
        hover_row = []
        for env in envs:
            # Average bias scores across all runs in this cell
            runs = s["environments"][env]["runs"]
            avg_biases = {bt: 0.0 for bt in bias_types}
            
            for run in runs:
                if "cognitive_bias" in run:
                    for bt in bias_types:
                        avg_biases[bt] += run["cognitive_bias"]["scores"][bt]
            
            # Normalize by number of runs
            n_runs = len(runs) if len(runs) > 0 else 1
            for bt in bias_types:
                avg_biases[bt] /= n_runs
            
            # Find dominant bias
            dominant_bias = max(avg_biases, key=avg_biases.get)
            dominant_score = avg_biases[dominant_bias] * 100
            
            row.append(dominant_score)
            hover_row.append(
                f"{s['topic']}<br>{env.capitalize()}<br>"
                f"Dominant: {dominant_bias.replace('_', ' ').title()}<br>"
                f"Score: {dominant_score:.1f}%"
            )
        
        z.append(row)
        hover_text.append(hover_row)
    
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[e.capitalize() for e in envs],
            y=topics,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorscale=[
                [0.0, "#1a1a2e"],
                [0.3, "#16213e"],
                [0.6, "#0f3460"],
                [1.0, "#58a6ff"],
            ],
            zmin=0, zmax=100,
            showscale=True,
            colorbar=dict(
                title=dict(text="Bias<br>Score %", font=dict(color=MUTED, size=10)),
                tickfont=dict(color=MUTED, size=9),
                thickness=12,
                len=0.8,
            ),
        )
    )
    
    fig.update_layout(
        **_base_layout(
            title=dict(text="Cognitive Bias Distribution", font=dict(size=13, color=MUTED)),
            xaxis=dict(
                tickfont=dict(color=TEXT),
                showgrid=False,
                side="top",
            ),
            yaxis=dict(
                tickfont=dict(color=TEXT),
                showgrid=False,
                autorange="reversed",
            ),
        )
    )
    return fig


# ── 7. Emotional Contagion Scatter ───────────────────────────────────────────

def plot_emotional_contagion(scenario_results: list) -> go.Figure:
    """Scatter plot showing emotional contagion: prompt anxiety vs response anxiety."""
    prompt_anxiety = []
    response_anxiety = []
    colors_list = []
    hover_texts = []
    
    for s in scenario_results:
        for env in ENV_ORDER:
            runs = s["environments"][env]["runs"]
            for run in runs:
                if "prompt_emotion" in run and "response_emotion" in run:
                    p_anx = run["prompt_emotion"]["scores"]["anxious"]
                    r_anx = run["response_emotion"]["scores"]["anxious"]
                    
                    prompt_anxiety.append(p_anx)
                    response_anxiety.append(r_anx)
                    colors_list.append(ENV_COLORS[env])
                    hover_texts.append(
                        f"{s['topic']} - {env.capitalize()}<br>"
                        f"Prompt anxiety: {p_anx:.2f}<br>"
                        f"Response anxiety: {r_anx:.2f}<br>"
                        f"Δ: {r_anx - p_anx:+.2f}"
                    )
    
    fig = go.Figure()
    
    # Add diagonal reference line (perfect contagion)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(color=MUTED, width=1, dash="dash"),
        name="Perfect contagion",
        hoverinfo="skip",
    ))
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=prompt_anxiety,
        y=response_anxiety,
        mode="markers",
        marker=dict(
            size=8,
            color=colors_list,
            opacity=0.7,
            line=dict(width=1, color=BG),
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        name="Runs",
    ))
    
    fig.update_layout(
        **_base_layout(
            title=dict(text="Emotional Contagion: Prompt → Response", font=dict(size=13, color=MUTED)),
            xaxis=dict(
                range=[0, 1],
                showgrid=True,
                gridcolor=GRID,
                tickfont=dict(color=MUTED),
                title=dict(text="Prompt Anxiety Score", font=dict(color=MUTED, size=10)),
            ),
            yaxis=dict(
                range=[0, 1],
                showgrid=True,
                gridcolor=GRID,
                tickfont=dict(color=MUTED),
                title=dict(text="Response Anxiety Score", font=dict(color=MUTED, size=10)),
            ),
            showlegend=True,
            legend=dict(
                font=dict(color=TEXT, size=10),
                bgcolor=PAPER_BG,
                bordercolor=GRID,
                borderwidth=1,
            ),
        )
    )
    return fig


# ── 8. Bias Distribution Bars ────────────────────────────────────────────────

def plot_bias_distribution(scenario_results: list) -> go.Figure:
    """Stacked bar chart showing cognitive bias distribution by environment."""
    bias_types = ["herd_mentality", "loss_aversion", "overconfidence", "neutral"]
    bias_labels = ["Herd/FOMO", "Loss Aversion", "Overconfidence", "Neutral"]
    bias_colors = [PURPLE, DANGER, WARN, SAFE]
    
    # Aggregate bias counts by environment
    env_bias_counts = {env: {bt: 0 for bt in bias_types} for env in ENV_ORDER}
    
    for s in scenario_results:
        for env in ENV_ORDER:
            runs = s["environments"][env]["runs"]
            for run in runs:
                if "cognitive_bias" in run:
                    top_bias_full = run["cognitive_bias"]["top_bias"]
                    # Map full label back to key
                    if "herd mentality" in top_bias_full.lower():
                        env_bias_counts[env]["herd_mentality"] += 1
                    elif "loss aversion" in top_bias_full.lower():
                        env_bias_counts[env]["loss_aversion"] += 1
                    elif "overconfidence" in top_bias_full.lower():
                        env_bias_counts[env]["overconfidence"] += 1
                    else:
                        env_bias_counts[env]["neutral"] += 1
    
    fig = go.Figure()
    
    for i, (bias_key, bias_label) in enumerate(zip(bias_types, bias_labels)):
        counts = [env_bias_counts[env][bias_key] for env in ENV_ORDER]
        fig.add_trace(go.Bar(
            x=[e.capitalize() for e in ENV_ORDER],
            y=counts,
            name=bias_label,
            marker=dict(color=bias_colors[i], opacity=0.85),
        ))
    
    fig.update_layout(
        **_base_layout(
            title=dict(text="Cognitive Bias Distribution by Environment", font=dict(size=13, color=MUTED)),
            xaxis=dict(showgrid=False, tickfont=dict(color=TEXT)),
            yaxis=dict(
                showgrid=True,
                gridcolor=GRID,
                tickfont=dict(color=MUTED),
                title=dict(text="Count", font=dict(color=MUTED, size=10)),
            ),
            barmode="stack",
            legend=dict(
                font=dict(color=TEXT, size=10),
                bgcolor=PAPER_BG,
                bordercolor=GRID,
                borderwidth=1,
            ),
        )
    )
    return fig
