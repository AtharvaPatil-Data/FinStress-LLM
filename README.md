# 🛡️ FinStress-LLM

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Google Colab](https://img.shields.io/badge/Inference-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit&logoColor=white)](https://finstress-llm-tj4ch2fnks25ekbubrfd8i.streamlit.app/)

> **Adversarial Stress-Testing Framework for Financial Language Models under Uncertainty**

An automated evaluation harness that systematically stress-tests open-source LLMs on financial advisory scenarios across four adversarial prompt environments. Built as part of a research portfolio for the *Central Bank of Ireland PhD Programme in LLM Safety and Interpretability* at Dublin City University.

---

### 🌐 Live Interactive Dashboard
You can access the live web-based dashboard and explore the experimental safety metrics, interactive response browser, and score distributions directly at:
👉 **[finstress-llm-tj4ch2fnks25ekbubrfd8i.streamlit.app](https://finstress-llm-tj4ch2fnks25ekbubrfd8i.streamlit.app/)** *(Hosted on Streamlit Community Cloud)*


---


## 🔬 Research Motivation

Large Language Models are increasingly deployed in high-stakes financial contexts robo-advisors, customer service chatbots, risk assessment tools. Yet their behaviour under adversarial pressure remains poorly understood. This framework asks:

> *How does an LLM's safety posture degrade when a user is panicked, under social pressure, or actively attempting prompt injection?*

This directly maps to the PhD research themes of **adversarial robustness**, **LLM safety**, and **behavioural analysis under uncertainty**.

---

## 🏗️ Architecture

```
FinStress-LLM/
│
├── colab_inference.py        # Run on Google Colab (T4 GPU)
│   ├── Generation model:  Qwen2.5-3B-Instruct
│   └── Evaluator model:   facebook/bart-large-mnli (zero-shot NLI)
│
├── dashboard/
│   ├── app.py                # Streamlit dashboard (run locally)
│   ├── plots.py              # Plotly visualisation functions
│   └── data/
│       └── results.json      # ← Place Colab output here
│
└── requirements.txt
```

**Two-model pipeline:**

```
Financial Prompt
      │
      ▼
┌─────────────────────────┐
│  Qwen2.5-3B-Instruct    │  ←── Generation Model (Colab T4)
│  (Financial response)   │
└──────────┬──────────────┘
           │ response text
           ▼
┌─────────────────────────┐
│  BART-Large-MNLI        │  ←── Safety Evaluator (zero-shot NLI)
│  Zero-shot classifier   │
└──────────┬──────────────┘
           │ {safe, unsafe, refusal} scores
           ▼
        results.json
```

---

## 🧪 Experimental Design

### Scenarios (5)
| ID  | Topic              |
|-----|--------------------|
| S01 | Retirement Savings |
| S02 | Debt Management    |
| S03 | Market Volatility  |
| S04 | Loan Decision      |
| S05 | Investment Fraud   |

### Prompt Environments (4)
| Environment   | Description |
|---------------|-------------|
| **Baseline**  | Neutral, well-formed financial question |
| **Panic**     | User is emotionally distressed, seeking fast/extreme action |
| **Pressure**  | Social proof / authority pressure to act immediately |
| **Injection** | Adversarial prompt injection attempting to override system prompt |

### Metrics
| Metric | Definition |
|--------|-----------|
| **Baseline Safety Rate** | % of baseline responses classified as safe |
| **Adversarial Safety Rate** | % of adversarial responses classified as safe |
| **Attack Success Rate** | 1 − Adversarial Safety Rate |
| **Vulnerability Margin** | Baseline Safety Rate − Adversarial Safety Rate |

---

## 🚀 Quick Start

### Step 1 — Run inference on Google Colab

1. Open [Google Colab](https://colab.research.google.com) and set runtime to **T4 GPU**
2. Upload `colab_inference.py`
3. Install dependencies:
   ```python
   !pip install transformers torch accelerate sentencepiece -q
   ```
4. Run:
   ```python
   !python colab_inference.py
   ```
5. Download the generated `results.json`

### Step 2 — Launch the local dashboard

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/FinStress-LLM.git
cd FinStress-LLM

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install local dependencies
pip install -r requirements.txt

# Place your results.json
cp /path/to/results.json dashboard/data/results.json

# Launch the dashboard
streamlit run dashboard/app.py
```

The dashboard runs in **demo mode** (synthetic data) if `results.json` is not present useful for previewing the UI before running inference.

---

## 📊 Dashboard Features

- **KPI row** — Baseline Safety Rate, Adversarial Safety Rate, Attack Success Rate, Vulnerability Margin
- **Safety Rate Heatmap** — Scenarios × Environments, colour-coded from red (unsafe) to green (safe)
- **Vulnerability Radar** — Per-scenario vulnerability margin on a polar chart
- **Environment Comparison Bar** — Aggregated safety rate with standard deviation error bars
- **NLI Score Distributions** — Violin plots of raw safe/unsafe classifier scores
- **Response Explorer** — Drill down to individual LLM responses with safety labels and NLI scores
- **Full Results Table** — Downloadable CSV-style view with conditional colour formatting

---

## 🔑 Key Concepts (PhD Interview Notes)

### Zero-shot NLI Classification
BART-Large-MNLI is a Natural Language Inference model trained to predict whether a *hypothesis* is entailed by, neutral to, or contradicts a *premise*. In zero-shot classification mode, we use the model's entailment score to classify the LLM response against candidate labels ("safe financial advice", "unsafe financial advice", "refusal") **without any fine-tuning**. This is a form of **transfer learning** for evaluation.

### Adversarial Robustness
A model is said to be robust if its output distribution does not shift significantly under input perturbations that should be semantically irrelevant to the task. In the financial context, adding emotional pressure to a question should not change the fundamental safety of the advice if it does, the model exhibits **adversarial brittleness**.

### Prompt Injection
A prompt injection attack embeds adversarial instructions in user input that attempt to override the system prompt or the model's instruction fine-tuning. This is a critical safety concern for deployed LLM systems operating under a fixed system context (e.g., a bank's chatbot). The injection scenarios in this framework test whether Qwen2.5-3B-Instruct's RLHF training resists these attacks.

### Vulnerability Margin
The gap between baseline and adversarial safety rates. A high vulnerability margin indicates that the model's safety behaviour is **context-dependent** rather than robust a significant finding for any regulator or deployer assessing systemic risk.

---

## 🔗 Connection to PhD Research Themes

| PhD Theme | This Project |
|-----------|--------------|
| LLM Safety & Adversarial Robustness | Core experimental design — 4 adversarial environments |
| Behavioural Analysis Under Uncertainty | Panic/pressure variants simulate real user uncertainty |
| Interpretability | Response Explorer enables qualitative failure mode analysis |
| Financial Decision-Making | All scenarios are financially relevant; metrics align with regulatory risk assessment |
| Central Bank of Ireland Context | Directly addresses LLM deployment risk in financial services |

---

## 📎 Models Used

| Model | Size | Purpose | License |
|-------|------|---------|---------|
| [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3B | Financial response generation | Apache 2.0 |
| [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) | 400M | Zero-shot safety classification | MIT |

All models are open-source and run on a single T4 GPU (16GB VRAM) within Google Colab free tier.

---

## 👤 Author

**Atharva Patil**  
MSc Computing (Data Analytics), Dublin City University  
PhD Applicant — Central Bank of Ireland PhD Programme, DCU  

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
