# Full-Stack TN Scheme Eligibility Predictor
## FastAPI + React / Tailwind Production Architecture

This project upgrades the original Streamlit prototype into a decoupled, production-grade full-stack web application:
- **Backend:** Python **FastAPI** serving high-speed RESTful inference with real-time SHAP explainability.
- **Frontend:** **React** + **Vite** + **Tailwind CSS** + **Lucide Icons** providing a modern, responsive civic-tech interface.
- **Machine Learning:** Existing MultiOutputClassifier (XGBoost) trained on 20 Tamil Nadu Government welfare schemes.

---

## 1. Project Directory Architecture

```
ML PROJECT/
│
├── backend/                              # Python FastAPI Application
│   ├── main.py                           # REST API routes (/predict, /schemes, /metrics, /health)
│   ├── predictor.py                      # Model loading, dummy encoding parity & SHAP explainability
│   ├── schemas.py                        # Pydantic input/output validation models
│   ├── test_api.py                       # Automated verification test script
│   └── requirements.txt                  # Python dependencies (fastapi, uvicorn, etc.)
│
├── frontend/                             # React + Tailwind Single Page Application
│   ├── index.html                        # Application HTML entry point
│   ├── package.json                      # Node dependencies & scripts
│   ├── vite.config.js                    # Vite configuration with API reverse proxy
│   ├── tailwind.config.js                # Custom Tamil Nadu civic color theme & typography
│   ├── postcss.config.js                 # PostCSS setup for Tailwind
│   └── src/
│       ├── main.jsx                      # React DOM mounting
│       ├── App.jsx                       # Main application shell & tab routing
│       ├── index.css                     # Tailwind directives & base styles
│       ├── api.js                        # Asynchronous fetch client for backend endpoints
│       └── components/
│           ├── Header.jsx                # Civic header with state logo & navigation tabs
│           ├── EligibilityForm.jsx       # 4-section citizen form with dynamic demo presets
│           ├── ResultsSection.jsx        # Eligibility cards, confidence meters & SHAP drivers
│           ├── SchemeDirectory.jsx       # Searchable & filterable 20-scheme catalog
│           ├── ModelMetrics.jsx          # Benchmark KPIs, comparison table & F1 breakdown
│           └── Footer.jsx                # Civic footer with official portal hyperlinks
│
├── scheme_model.pkl                      # Trained MultiOutputClassifier (20 XGBoost estimators)
├── model_feature_columns.pkl             # List of exact 36 one-hot dummy encoded features
├── scheme_rules_full.xlsx                # Official criteria rules for all 20 schemes
├── model_comparison.csv                  # RF vs LR vs XGBoost benchmark data
├── per_scheme_f1.csv                     # Individual test F1 scores
└── app.py                                # Original Streamlit app (kept intact as fallback)
```

---

## 2. API Endpoints Specification

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status and root service metadata |
| `GET` | `/api/health` | Service health status and loaded model info |
| `GET` | `/api/schemes` | Returns catalog of 20 TN schemes with rules & criteria |
| `GET` | `/api/metrics` | Model comparison table and per-scheme F1 test scores |
| `POST` | `/api/predict` | Evaluates 19 citizen features and returns ranked eligible schemes with SHAP factors |

---

## 3. How to Run Locally

### Prerequisites
1. **Python 3.10+** (already configured in `venv/`)
2. **Node.js 18+ & npm** ([Download from nodejs.org](https://nodejs.org) if not yet installed)

---

### Step A: Start the FastAPI Backend
Open a PowerShell terminal:
```powershell
cd "C:\Users\SUJANTH\OneDrive\Desktop\ML PROJECT"
venv\Scripts\uvicorn.exe backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger API Docs will be available at: **http://127.0.0.1:8000/docs**
- Health status: **http://127.0.0.1:8000/api/health**

---

### Step B: Start the React Frontend
Open a **second** PowerShell terminal:
```powershell
cd "C:\Users\SUJANTH\OneDrive\Desktop\ML PROJECT\frontend"
npm install
npm run dev
```
- Open your browser at: **http://localhost:3000** (or the port Vite displays)

---

## 4. Production Deployment Recommendations

| Component | Recommended Free / Cloud Host | How to Deploy |
|---|---|---|
| **Frontend (React)** | **Vercel** / **Netlify** | Connect GitHub repo, set root directory to `frontend`, build command `npm run build`, output `dist`. |
| **Backend (FastAPI)** | **Render** / **Railway** / **Fly.io** | Use Docker or Python runtime with start command `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`. |
