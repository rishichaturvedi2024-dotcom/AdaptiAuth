# AdaptiAuth

### Risk-Adaptive Continuous Authentication Framework Using Multi-Modal Biometrics and Explainable Trust Scoring

---

## Overview

AdaptiAuth replaces one-time, point-in-login authentication with **continuous, explainable trust evaluation** that persists for the entire session. It fuses three biometric signal layers into a single Trust Score (0–1), explains that score with SHAP, and feeds it into a zero-trust policy engine that decides whether to leave the session alone, force step-up authentication, or terminate it.

## Architecture

### Three Signal Layers

| Layer | Signal | Output |
|-------|--------|--------|
| **Layer 1 — Deep Facial Verification** | CNN-based facial embedding (128-dim) after 3D face alignment | Identity verification score (cosine similarity vs. enrolled template) |
| **Layer 2 — Dual-Layer Liveness Detection** | Remote photoplethysmography (rPPG) + deep-learning presentation-attack detector | Physiological liveness confidence + spoof probability → fused liveness sub-score |
| **Layer 3 — Behavioral & Contextual Monitoring** | Keystroke dynamics, mouse movement, device fingerprint, geo-velocity, login time, network context | Behavioral consistency score against rolling session baseline |

### Fusion + Decisioning

- **Adaptive Trust-Scoring Engine** — Dynamically weighted fusion of the three layers into one continuously updated Trust Score (0–1).
- **Explainable AI Layer** — SHAP-based, per-feature attributions for every Trust Score, for audit and forensics.
- **Zero-Trust Policy Engine** — Maps Trust Score to a risk tier and an automated response:

| Trust Score | Risk Level | Response |
|-------------|------------|----------|
| 0.80 – 1.00 | Low | Uninterrupted access |
| 0.50 – 0.79 | Medium | FIDO2 / WebAuthn step-up authentication |
| 0.00 – 0.49 | High | Terminate session + generate SOC security alert |

### Key Novelties

1. **Continuous** (not point-in-time) evaluation throughout the session
2. **rPPG + deep-learning liveness** fused together (not heuristic blink/head-turn checks)
3. **SHAP explainability** applied to the fused, continuous, multi-modal trust score
4. **Automated risk-tiered policy response**, reducing reliance on static OTPs/tokens
5. **On-device-only federated updates** for the continuous monitoring loop (privacy-preserving retraining; no raw biometric data leaves the device)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend / ML services | Python 3.11+, FastAPI |
| Facial embeddings | MobileFaceNet-style CNN, OpenCV, face-alignment |
| rPPG | Chrominance-based signal processing pipeline |
| Presentation-attack detection | Small CNN classifier |
| Behavioral biometrics | LSTM/Transformer sequence model |
| Trust fusion | Weighted fusion module (rule-based, swappable for learned) |
| Explainability | SHAP |
| Policy engine | Deterministic state machine |
| Frontend | React (Vite) |
| Data/session store | SQLite (swap-in note for PostgreSQL in production) |
| Auth step-up | WebAuthn (py_webauthn + @simplewebauthn/browser) |

## Project Structure

```
adaptiauth/
├── backend/                  # FastAPI backend
│   └── app/
│       ├── main.py           # Application entry point
│       ├── api/
│       │   └── routes/       # API route handlers
│       │       ├── auth.py         # Authentication endpoints
│       │       ├── session.py      # Session management
│       │       ├── trust.py        # Trust score queries
│       │       └── dashboard.py    # SOC dashboard endpoints
│       ├── core/
│       │   ├── config.py     # App configuration
│       │   ├── database.py   # SQLite/database layer
│       │   └── security.py   # Security utilities
│       ├── models/
│       │   └── schemas.py    # Pydantic models
│       └── services/         # Business logic services
│           ├── facial.py           # Layer 1 service
│           ├── liveness.py         # Layer 2 service
│           ├── behavioral.py       # Layer 3 service
│           ├── trust_engine.py     # Trust fusion engine
│           ├── explainability.py   # SHAP explainability
│           └── policy_engine.py    # Zero-trust policy engine
├── ml/                       # ML models and pipelines
│   ├── facial/               # Facial verification models
│   ├── liveness/             # Liveness detection models
│   ├── behavioral/           # Behavioral biometrics models
│   ├── fusion/               # Trust fusion logic
│   └── explainability/       # SHAP integration
├── frontend/                 # React frontend (Vite)
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Page views
│   │   ├── services/         # API client services
│   │   └── hooks/            # Custom React hooks
│   └── public/
├── docs/                     # Documentation
│   ├── architecture.md       # Architecture diagrams
│   └── ip_alignment.md       # IP/novelty traceability (Phase 10)
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests per module
│   └── integration/          # Integration/end-to-end tests
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
└── README.md                 # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- A webcam (for facial/liveness demo)

### 1. Start the Backend

```bash
cd adaptiauth
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

### 2. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

### 3. Run the Demo End-to-End

1. Open your browser to the Vite dev server URL (usually `http://localhost:5173`).
2. Click **Launch Login Demo**.
3. Allow webcam access. You will see simulated telemetry for Layer 1 & 2 (Webcam) and Layer 3 (Behavioral - move mouse/type to see it update).
4. In a separate tab, open **View SOC Dashboard** from the homepage to see the active session, continuous risk scoring, and live SHAP attributions updating over time.

### Run Tests

```bash
pytest tests/ -v
```

### Known Limitations (Prototype)
- **Biometric Models**: Facial embeddings use `facenet-pytorch`, but liveness (rPPG/PAD) and behavioral models are currently **stubs** that return simulated scores.
- **WebAuthn**: The WebAuthn step-up flow is a backend policy stub, not fully wired to `@simplewebauthn/browser` in this demo.
- **Explainability**: SHAP is fully wired to the fusion engine, but relies on the mocked models for feature inputs.
- **Database**: Active sessions and telemetry are held in memory/mocked instead of a persistent SQLite DB.

## Development Status

> [!WARNING]
> This is a **functional prototype**, not a production security product.
> Security-relevant simplifications (stub models, demo WebAuthn, no production anti-spoofing dataset) are clearly flagged in code comments.

See `docs/architecture.md` for the full system diagram.

## License

Proprietary — all rights reserved. See IP documentation in `docs/ip_alignment.md`.
