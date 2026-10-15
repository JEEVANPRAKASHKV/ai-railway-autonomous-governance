# 🚄 AI-Powered Intelligent Railway Travel & Autonomous AI Governance Platform

> **Enterprise Railway AI Ecosystem** — Dual-Plane Passenger Intelligence & Autonomous AI Governance Platform  
> **Repository:** https://github.com/JEEVANPRAKASHKV/ai-railway-autonomous-governance  
> **Architect:** Jeevan Prakash K V

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-railway-governance.streamlit.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?logo=fastapi)](http://localhost:8000/docs)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests Passed](https://img.shields.io/badge/tests-35%20passed-success)](tests/)
[![Architecture](https://img.shields.io/badge/Architecture-Dual--Plane%20AI-blueviolet)](#-architecture)

---

### 🌐 Live Public Deployment
> **🚀 Access Live Control Tower & Passenger AI:** **[https://ai-railway-governance.streamlit.app](https://ai-railway-governance.streamlit.app)**  
> *(No local installation or API keys required — full dual-plane functionality and autonomous governance telemetry enabled out of the box).*

---

## 📋 Table of Contents

1. [Live Public Deployment](#-live-public-deployment)
2. [Platform Overview](#-platform-overview)
3. [Quick Start — 3 Steps](#-quick-start--3-steps)
4. [Cloud Deployment Guide (Streamlit Cloud & Render)](#-cloud-deployment-guide-streamlit-community-cloud--render)
5. [Architecture](#-architecture)
6. [7 Passenger AI Modules](#-7-passenger-ai-modules)
7. [8-Pillar AI Governance Engine](#-8-pillar-ai-governance-engine)
8. [Dual-Mode LLM Resilience](#-dual-mode-llm-resilience-zero-crash-guarantee)
9. [Running Without Azure OpenAI (Standalone Mode)](#-running-without-azure-openai-standalone-mode)
10. [Chaos Engineering Lab](#-chaos-engineering-lab)
11. [API Reference](#-api-reference)
12. [Project Structure](#-project-structure)
13. [Technology Stack](#-technology-stack)

---

## 🎯 Platform Overview

This platform is an enterprise-grade **dual-plane AI ecosystem** engineered for intelligent railway passenger services and real-time autonomous AI governance:

| Plane | Purpose | Entry |
|---|---|---|
| **Passenger Experience** | 7 intelligent AI services for Indian Railways travelers | Streamlit Tabs 1–7 · FastAPI `/api/v1/*` |
| **AI Governance Control Tower** | Autonomous "AI monitoring AI" cockpit — 8 governance pillars | Streamlit Tab 8 · FastAPI `/api/v1/governance/*` |

### Platform Golden Rules

| Rule | Implementation |
|---|---|
| **Self-Contained** | Deterministic synthetic data generator (seed=42) produces all 10 datasets · Models train in-repo |
| **Zero-Crash LLM** | Dual-mode router: Azure OpenAI primary → rule-based fallback (auto-switches, no code change) |
| **Single-Command Launch** | `python run_platform.py` — validates, generates data, trains models, starts both servers |
| **Embedded Governance** | `@governance_hook` decorator on every module · telemetry bus · autonomous supervisor agent |

---

## ⚡ Quick Start — 3 Steps

### Prerequisites
- Python 3.10 or higher
- ~2 GB disk space (for ML models and synthetic data)

```bash
# 1. Clone the repository
git clone https://github.com/JEEVANPRAKASHKV/ai-railway-autonomous-governance.git
cd ai-railway-autonomous-governance

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the full platform (one command)
python run_platform.py
```

The launcher automatically:
- ✅ Generates 10 synthetic datasets (~400K rows total) if not present
- ✅ Trains 5 ML models (XGBoost, LightGBM) with SHAP explainers if not present
- ✅ Starts FastAPI backend on **http://localhost:8000**
- ✅ Launches Streamlit Control Tower on **http://localhost:8501**

> **No external API keys required.** The platform functions out-of-the-box in standalone intelligent fallback mode. See [Running Without Azure OpenAI (Standalone Mode)](#-running-without-azure-openai-standalone-mode).

---

## ☁️ Cloud Deployment Guide (Streamlit Community Cloud & Render)

The repository features a self-contained root deployment entrypoint ([`streamlit_app.py`](streamlit_app.py)) that launches both the **FastAPI Governance Backend** (background daemon on `127.0.0.1:8000`) and the **Streamlit Passenger & Governance Portal** inside a single cloud container.

### Option A: Streamlit Community Cloud (Recommended — 100% Free)

[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
2. Click **"New app"** and configure:
   - **Repository:** `JEEVANPRAKASHKV/ai-railway-autonomous-governance`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL:** Customize your subdomain (e.g. `ai-railway-governance.streamlit.app`)
3. *(Optional)* Click **Advanced Settings** → **Secrets** if you wish to attach live Azure OpenAI credentials:
   ```toml
   AZURE_OPENAI_API_KEY = "your-api-key"
   AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
   AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
   AZURE_OPENAI_API_VERSION = "2024-02-15-preview"
   ```
   > **Note:** If no secrets are provided, the platform automatically boots into **Standalone Intelligent Rule-Based Fallback Mode** with 100% feature availability and zero crashes.
4. Click **"Deploy!"** — Within 60 seconds, your dual-plane platform will be live globally.

---

### Option B: Render (Web Service)

1. Sign in to **[dashboard.render.com](https://dashboard.render.com/)** and click **New +** → **Web Service**.
2. Connect your GitHub repository: `JEEVANPRAKASHKV/ai-railway-autonomous-governance`.
3. Set the build and start settings:
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
4. *(Optional)* Add Environment Variables under the **Environment** tab:
   - `BACKEND_URL`: `http://127.0.0.1:8000`
   - `PYTHONUNBUFFERED`: `1`
5. Click **Create Web Service**.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PASSENGER EXPERIENCE PLANE                          │
│                                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ M1:Ticket│  │M2:Trains │  │ M3:Delay │  │M4:Planner│  │ M5:Fare  │ │
│  │Predictor │  │Recommend │  │Predictor │  │ (Agentic)│  │Predictor │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
│  ┌──────────┐  ┌──────────┐                                             │
│  │M6:Support│  │M7:Fraud  │  ← All wrapped with @governance_hook        │
│  │Assistant │  │Detector  │                                             │
│  └──────────┘  └──────────┘                                             │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │ TelemetryEvent (latency, schema, confidence)
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              GOVERNANCE TELEMETRY BUS (in-memory, 5000 events)          │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  AI GOVERNANCE CONTROL TOWER (8 Pillars)                │
│                                                                         │
│  [1] Health Monitor    [2] Drift Detector    [3] Bias Auditor           │
│      SLA · P95 · CB        KS-test · PSI         DIR · DPD · EOD       │
│                                                                         │
│  [4] Hallucination     [5] RCA Engine        [6] Lifecycle Manager      │
│      Monitor               5-Why · DAG           Canary · Rollback      │
│      Faithfulness                                                        │
│                                                                         │
│  [7] Report Generator  [8] Supervisor Agent                             │
│      JSON · Markdown       Autonomous loop ← polls all 8 pillars        │
│                            HITL Gate · Chaos Injection                  │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              DUAL-MODE LLM ROUTER (core/llm_router.py)                  │
│                                                                         │
│  Primary: Azure OpenAI GPT-4o-mini ──────────────── 15s timeout        │
│      ↓ (if unavailable / timeout)                                       │
│  Fallback: Rule-Based Engine ─── TF-IDF RAG · Regex · Greedy Search    │
│                                  Zero crash · Full functionality        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow (Mermaid)

```mermaid
flowchart TD
    U([User]) -->|HTTP Request| ST[Streamlit :8501]
    ST -->|REST API| FA[FastAPI :8000]
    FA --> GH[@governance_hook decorator]
    GH --> DQ[Data Quality Gate]
    GH --> ML[ML Model / LLM Router]
    GH --> TB[Telemetry Bus]
    ML -->|Azure OpenAI| AZ[Azure GPT-4o-mini]
    ML -->|Fallback| FB[Rule-Based Engine]
    TB --> SA[Supervisor Agent]
    SA --> HM[Health Monitor]
    SA --> DD[Drift Detector]
    SA --> BA[Bias Auditor]
    SA --> HLM[Hallucination Monitor]
    SA --> RCA[RCA Engine]
    SA --> LM[Lifecycle Manager]
    SA --> RG[Report Generator]
    SA -->|HITL Gate| OP([Human Operator])
```

---

## 🤖 7 Passenger AI Modules

### Module 1 — Intelligent Ticket Confirmation Predictor
**Endpoint:** `POST /api/v1/predict/ticket`

Predicts waitlist/RAC clearance probability using XGBoost with full SHAP explainability.

- **Input:** Train ID, class, quota, WL number, days to departure, booking history
- **Output:** Confirmation probability, verdict (Likely Confirmed / Borderline / Likely WL), berth recommendation, SHAP feature contributions, governance badge
- **Governance:** Rolling accuracy monitor, auto-retrain trigger at accuracy < 0.80

---

### Module 2 — Smart Train Recommendation Engine
**Endpoint:** `POST /api/v1/recommend/trains`

Multi-factor persona-based scoring with built-in fairness auditing.

- **Scoring Formula:** `Score = w₁×(1/Fare) + w₂×Punctuality + w₃×Availability + w₄×Duration + w₅×Comfort + w₆×PersonaMatch`
- **Personas:** Budget (price-first) · Business (speed-first) · Family (availability-first) · Senior · Tourist
- **Governance:** Disparate Impact Ratio (DIR) — flags if budget trains under-represented in top-3

---

### Module 3 — Train Delay & Disruption Predictor
**Endpoint:** `POST /api/v1/predict/delay`

LightGBM delay forecasting with confidence intervals and proactive alternative routing.

- **Output:** Predicted delay (minutes), confidence interval, congestion level, alternative routes (if delay > 60 min)
- **Governance:** KS-test + PSI drift monitoring on congestion features. PSI > 0.25 → P0-Critical + retrain

---

### Module 4 — Agentic AI Travel Planner *(Human-in-the-Loop)*
**Endpoints:** `POST /api/v1/agent/plan` · `POST /api/v1/agent/confirm` · `POST /api/v1/agent/reject`

Multi-tool AI agent with mandatory Human-in-the-Loop booking gate.

**Agent Tools:** Route Search → Fare Estimator → WL Predictor → Delay Checker → Itinerary Optimizer → **HITL Gate** (blocks until human confirms)

- **Governance:** Prompt injection scanner, loop detection (>3 iterations → abort), full execution trace audit

---

### Module 5 — Dynamic Fare Prediction Engine
**Endpoint:** `POST /api/v1/predict/fare`

XGBoost 30-day fare trajectory with optimal booking window identification.

- **Output:** Current fare, 30-day forecast curve, optimal booking day, demand level (Low/Medium/High/Surge), SHAP fare drivers
- **Governance:** MAPE tracker + anomaly flag if predicted fare > 2.5× or < 0.5× base fare

---

### Module 6 — Multilingual Customer Support Assistant
**Endpoint:** `POST /api/v1/support/query`

RAG pipeline over IRCTC policies with multilingual support and hallucination monitoring.

**Pipeline:** Query → Language Detection → TF-IDF Retrieval (railway_faq.txt / irctc_policies.txt) → LLM/Fallback Generation → **Faithfulness Check** → Response

- **Languages:** English · Hindi · Hinglish (auto-detected)
- **Governance:** Faithfulness score (TF-IDF cosine vs. retrieved context). Score < 0.35 → quarantine + fallback

---

### Module 7 — Fraud, Security & Bot Detection
**Endpoint:** `POST /api/v1/security/score`

XGBoost real-time session risk scorer with Tatkal bot protection.

- **Risk Features:** Requests/minute, PNR attempt velocity, Tatkal window flag, CAPTCHA failures, bulk booking pattern
- **Output:** Fraud risk score (0–100), risk category (Safe/Suspicious/High Risk/Bot Detected), mitigation dossier
- **Governance:** Live confusion matrix tracker, threat model rule updater

---

## 🛡️ 8-Pillar AI Governance Engine

> *"AI monitoring AI"* — The governance layer autonomously supervises all 7 passenger modules with zero human intervention required.

### Pillar 1 — System Health Monitor (`health_monitor.py`)
- Per-module latency percentiles: P50 / P95 / P99
- **Circuit Breaker** pattern: CLOSED → OPEN (after 5 consecutive failures) → HALF-OPEN (recovery probe after 60s)
- SLA compliance tracking: P95 < 3000ms, error rate < 5%
- Publishes `SLA_BREACH` events to telemetry bus

### Pillar 2 — Data Quality Gate (`data_quality_gate.py`)
- Pre-inference schema validation against `feature_columns.json`
- Numeric range bounds checking (e.g., WL number ∈ [1, 500])
- Z-score anomaly detection (z > 3.5 vs. training baselines)
- Modules still serve with warning badge — no hard block

### Pillar 3 — Statistical Drift Detector (`drift_detector.py`)
- **Kolmogorov-Smirnov (KS) test** per numeric feature vs. training baseline
- **Population Stability Index (PSI):** 0.0–0.10 Stable · 0.10–0.20 Warning · >0.20 Critical
- Rolling 200-sample windows; prediction distribution drift detection
- Triggers retraining queue on drift detection

### Pillar 4 — Bias & Fairness Auditor (`bias_auditor.py`)
- **Disparate Impact Ratio (DIR)** — EEOC 4/5ths rule (DIR < 0.8 → adverse impact)
- **Demographic Parity Difference (DPD)** — positive rate gap across groups
- **Equal Opportunity Difference (EOD)** — TPR gap when ground truth available
- Protected attributes: gender, age group, class preference, booking method, city tier

### Pillar 5 — Root Cause Analysis Engine (`rca_engine.py`)
- Event-driven fault tree construction from telemetry stream
- **Automated 5-Why reasoning** chains per incident type
- Causal dependency graph traversal
- Prioritized remediation playbooks (IMMEDIATE / HIGH / MEDIUM)
- Incident lifecycle: Create → Correlate → Analyze → Resolve

### Pillar 6 — Model Lifecycle Manager (`lifecycle_manager.py`)
- Full version registry (staging → canary → production → archived)
- **Canary deployment** with configurable traffic split (default 10%)
- **Champion/Challenger A-B testing** framework
- **Automatic rollback** triggered by governance events (SLA_BREACH, DRIFT_DETECTED)
- Retraining queue management

### Pillar 7 — Automated Report Generator (`report_generator.py`)
- Generates comprehensive audit reports on demand
- **JSON** (machine-readable API response) + **Markdown** (human-readable, Git-friendly)
- Executive summary · risk assessment · critical flags · ranked recommendations
- Saved to `governance/reports/` with timestamp-based IDs

### Pillar 8 — Autonomous Supervisor Agent (`supervisor_agent.py`)
- Background polling loop (every 30 seconds) orchestrating all 7 pillars
- **HITL (Human-in-the-Loop) approval gates** for high-risk actions (promotions, mass rollbacks)
- Chaos injection for resilience testing: `latency_spike` · `error_surge` · `drift_simulation`
- Unified dashboard state API powering the Streamlit Control Tower
- Governance action audit trail (last 500 actions)

---

## 🔀 Dual-Mode LLM Resilience (Zero-Crash Guarantee)

```
┌──────────────────────────────────────────────────────────────┐
│                    LLM Router Decision Tree                  │
│                                                              │
│  Azure OpenAI credentials in .env?                          │
│          │                                                   │
│    YES ──┼── Initialize AzureChatOpenAI (GPT-4o-mini)       │
│          │   ├── 15-second timeout with 2 retries           │
│          │   └── On failure → auto-fallback (transparent)   │
│          │                                                   │
│    NO  ──┼── Activate Rule-Based Fallback Engine            │
│              ├── M4 Travel Planner: Regex + greedy search   │
│              │   over trains.csv                            │
│              ├── M6 Support: TF-IDF cosine similarity over  │
│              │   railway_faq.txt (top-1 answer)             │
│              └── Governance RCA: Decision-tree diagnosis    │
└──────────────────────────────────────────────────────────────┘
```

**The mode switch is completely transparent** — the same API endpoints, same response schemas, and unified UI. Operators see `Mode: Rule-Based Fallback 🔧` in the sidebar when running standalone.

---

## 🚫 Running Without Azure OpenAI (Standalone Mode)

**No external cloud API required. Standalone execution is supported natively.**

1. Leave `.env` as-is (placeholder values trigger fallback mode automatically)
2. Run `python run_platform.py`
3. Streamlit sidebar shows: `🔧 Rule-Based Fallback` indicator

All 7 modules and all 8 governance pillars work identically. The only difference:
- **M4 Planner:** Uses regex + greedy itinerary builder instead of GPT-4o-mini
- **M6 Support:** Returns TF-IDF top-1 answer instead of LLM-generated response

All ML predictions (M1, M2, M3, M5, M7) are **always local** — never use any cloud API.

---

## 🌪️ Chaos Engineering Lab

The platform includes a built-in chaos simulator to demonstrate autonomous governance resilience:

| Chaos Type | What it injects | Expected governance response |
|---|---|---|
| `latency_spike` | Simulates 5000ms responses | Circuit breaker → SLA_BREACH event → RCA incident created |
| `error_surge` | Injects 8 consecutive failures | Circuit opens → OPEN state → auto-recovery attempt after 60s |
| `drift_simulation` | Publishes synthetic drift event | Drift detected → retraining queued → supervisor logs action |

**How to use:**
1. Open Streamlit → **🧪 Chaos Engineering Lab** page
2. Select target module and chaos type
3. Click **Inject Chaos**
4. Navigate to **📊 Platform Health** and **🚨 Incidents & RCA** to see governance response

The entire loop (inject → detect → alert → remediate) completes in **< 5 seconds**.

---

## 📡 API Reference

All endpoints available at **http://localhost:8000/docs** (Swagger UI)

### Passenger Experience APIs

| Method | Endpoint | Module |
|---|---|---|
| POST | `/api/v1/predict/ticket` | M1 — Ticket Predictor |
| POST | `/api/v1/recommend/trains` | M2 — Train Recommender |
| POST | `/api/v1/predict/delay` | M3 — Delay Predictor |
| POST | `/api/v1/agent/plan` | M4 — Travel Planner |
| POST | `/api/v1/agent/confirm` | M4 — HITL Confirm |
| POST | `/api/v1/agent/reject` | M4 — HITL Reject |
| POST | `/api/v1/predict/fare` | M5 — Fare Predictor |
| POST | `/api/v1/support/query` | M6 — Support Assistant |
| POST | `/api/v1/security/score` | M7 — Fraud Detector |

### Governance Control Tower APIs (30+ endpoints)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/governance/supervisor/dashboard` | Full dashboard state |
| POST | `/api/v1/governance/supervisor/report` | Generate audit report |
| POST | `/api/v1/governance/supervisor/rollback` | Manual model rollback |
| POST | `/api/v1/governance/supervisor/chaos` | Chaos injection |
| GET | `/api/v1/governance/health/platform` | Platform health |
| GET | `/api/v1/governance/drift/all` | Drift analysis (all models) |
| GET | `/api/v1/governance/bias/audit/all` | Fairness audit |
| GET | `/api/v1/governance/hallucination/stats` | LLM faithfulness |
| GET | `/api/v1/governance/incidents/open` | Open incidents |
| GET | `/api/v1/governance/lifecycle/platform` | Model lifecycle |
| GET | `/api/v1/governance/telemetry` | Live telemetry stream |

---

## 📁 Project Structure

```
ai-railway-autonomous-governance/
│
├── run_platform.py                     # Single-command launcher
├── .env.example                        # Config template (no keys needed)
├── requirements.txt                    # All pinned dependencies
├── README.md                           # This file
│
├── data/
│   ├── generate_synthetic_data.py      # Deterministic generator (seed=42)
│   ├── raw/                            # Generated datasets (10 files, ~400K rows)
│   └── baseline_distributions.pkl     # Feature stats for drift detection
│
├── models/
│   ├── train_all_models.py             # Trains 5 ML models with SHAP
│   ├── artifacts/                      # Serialized models + SHAP explainers
│   └── registry/model_registry.json   # Version table with metrics
│
├── core/
│   ├── llm_router.py                   # Dual-mode LLM router
│   ├── fallback_engine.py              # Rule-based fallback (TF-IDF + regex)
│   ├── config.py                       # PlatformSettings (pydantic)
│   └── schemas.py                      # Shared Pydantic V2 schemas
│
├── governance/                         # 8-pillar governance engine
│   ├── telemetry_bus.py                # Thread-safe ring buffer (5000 events)
│   ├── hooks.py                        # @governance_hook decorator
│   ├── data_quality_gate.py            # Pre-inference schema validator
│   ├── health_monitor.py               # SLA + circuit breaker (Pillar 1)
│   ├── drift_detector.py               # KS-test + PSI (Pillar 2+3)
│   ├── bias_auditor.py                 # DIR + DPD + EOD (Pillar 4)
│   ├── hallucination_monitor.py        # LLM faithfulness (Pillar 4)
│   ├── rca_engine.py                   # 5-Why + fault tree (Pillar 5)
│   ├── lifecycle_manager.py            # Canary + rollback (Pillar 6)
│   ├── report_generator.py             # JSON + Markdown reports (Pillar 7)
│   ├── supervisor_agent.py             # Autonomous meta-agent (Pillar 8)
│   └── reports/                        # Generated audit reports
│
├── modules/                            # 7 ML service modules
│   ├── m1_ticket_predictor/service.py
│   ├── m2_train_recommender/service.py
│   ├── m3_delay_predictor/service.py
│   ├── m4_travel_planner/agent.py
│   ├── m5_fare_predictor/service.py
│   ├── m6_support_assistant/service.py
│   └── m7_fraud_detector/service.py
│
├── backend/
│   ├── main.py                         # FastAPI app entry point
│   ├── middleware.py                   # Telemetry capture middleware
│   ├── dependencies.py                 # Shared FastAPI dependencies
│   └── routers/                        # 8 router files (M1–M7 + governance)
│
├── frontend/
│   └── app.py                          # Streamlit Control Tower (12 pages)
│
└── tests/
    ├── test_data_generator.py
    ├── test_modules.py
    └── test_governance.py
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Web Framework** | FastAPI 0.111 · Uvicorn · Pydantic V2 |
| **Frontend** | Streamlit 1.35 · Plotly · Premium dark CSS |
| **ML Models** | XGBoost 2.0 · LightGBM 4.3 · scikit-learn 1.5 |
| **Explainability** | SHAP 0.45 (TreeExplainer per model) |
| **Drift Detection** | SciPy KS-test · PSI (Population Stability Index) |
| **LLM (Primary)** | Azure OpenAI GPT-4o-mini via LangChain |
| **LLM (Fallback)** | TF-IDF cosine similarity · regex rule engine |
| **Agentic Framework** | LangGraph · LangChain tools |
| **Governance Bus** | Custom thread-safe in-memory ring buffer |
| **Data** | Pandas · NumPy · deterministic synthetic generator |
| **Config** | pydantic-settings · python-dotenv |

---

## 📊 Synthetic Dataset Summary

| Dataset | Rows | Description |
|---|---|---|
| `trains.csv` | 500 | Indian railway routes, classes, quota types |
| `stations.csv` | 200 | Stations with zone, tier, baseline congestion |
| `pnr_history.csv` | 50,000 | WL/RAC/Confirmed records with final status |
| `pricing_history.csv` | 100,000 | Fare-date-class records with dynamic curves |
| `user_profiles.csv` | 5,000 | 6 personas: budget, premium, student, senior, business, family |
| `congestion_logs.csv` | 73,000 | 365 days × 200 stations congestion index |
| `delay_logs.csv` | 200,000 | Historical delay per train per station |
| `fraud_events.csv` | 10,000 | Labeled sessions (80% legit, 5% bot, 15% suspicious) |
| `irctc_policies.txt` | ~3,000 lines | Cancellation, refund, Tatkal, waitlist policies |
| `railway_faq.txt` | 500 Q&A | PNR, refund, booking queries |

All data generated deterministically with `RANDOM_SEED=42` — reproducible across all machines.

---

## 🤝 Contributing / Extending

The platform is modular by design. To add a new AI module:

1. Create `modules/m8_new_module/service.py`
2. Add `@governance_hook(module_id="m8_new_module")` to the service function
3. Create `backend/routers/new_module.py` and register in `backend/main.py`
4. Add a Streamlit page in `frontend/app.py`

The governance layer automatically picks up the new module's telemetry — **no governance code changes needed**.

---

*AI-Powered Intelligent Railway Travel & Autonomous AI Governance Platform — Enterprise Reference Architecture.*