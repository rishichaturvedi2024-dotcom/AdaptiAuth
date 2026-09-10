# AdaptiAuth — System Architecture

## High-Level Block Diagram

```mermaid
flowchart TB
    subgraph SENSING["📷 Sensing Layer"]
        CAM["Webcam Feed"]
        KBD["Keyboard Events"]
        MOUSE["Mouse Events"]
        CTX["Context Signals<br/>(Device, IP, Time)"]
    end

    subgraph ENCODERS["🧠 Per-Layer Encoders"]
        subgraph L1["Layer 1: Deep Facial Verification"]
            FD["Face Detection<br/>(MTCNN / RetinaFace)"]
            FA["3D Face Alignment"]
            FE["CNN Embedding<br/>(MobileFaceNet-style)<br/>→ 128-dim vector"]
            CS["Cosine Similarity<br/>vs Enrolled Template"]
            FD --> FA --> FE --> CS
        end

        subgraph L2["Layer 2: Dual-Layer Liveness"]
            RPPG["rPPG Extraction<br/>(Chrominance-based)"]
            PAD["Presentation Attack<br/>Detector (CNN)"]
            LF["Liveness Fusion<br/>(rPPG conf × PAD conf)"]
            RPPG --> LF
            PAD --> LF
        end

        subgraph L3["Layer 3: Behavioral & Context"]
            KD["Keystroke Dynamics<br/>Feature Extraction"]
            MM["Mouse Movement<br/>Feature Extraction"]
            DF["Device Fingerprint"]
            GV["Geo-Velocity Check"]
            BM["Sequence Model<br/>(LSTM/Transformer)"]
            KD --> BM
            MM --> BM
            DF --> BM
            GV --> BM
        end
    end

    subgraph FUSION["⚖️ Adaptive Trust-Scoring Engine"]
        DW["Dynamic Weight<br/>Calculator"]
        TS["Trust Score<br/>(0.0 – 1.0)"]
        DW --> TS
    end

    subgraph XAI["🔍 Explainability Layer"]
        SHAP["SHAP Explainer"]
        ATTR["Per-Feature<br/>Attributions"]
        SHAP --> ATTR
    end

    subgraph POLICY["🛡️ Zero-Trust Policy Engine"]
        SM["Risk-Tier<br/>State Machine"]
        subgraph RESPONSES["Automated Responses"]
            LOW["✅ Low Risk<br/>(0.80–1.00)<br/>Uninterrupted Access"]
            MED["⚠️ Medium Risk<br/>(0.50–0.79)<br/>WebAuthn Step-Up"]
            HIGH["🚨 High Risk<br/>(0.00–0.49)<br/>Session Kill + SOC Alert"]
        end
        SM --> LOW
        SM --> MED
        SM --> HIGH
    end

    subgraph DASHBOARD["📊 Audit & SOC Dashboard"]
        TSC["Trust Score<br/>Timeline Chart"]
        SHV["SHAP Explanation<br/>View"]
        AL["Alert Log"]
        SL["Session List"]
    end

    subgraph CONTINUOUS["🔄 Continuous Monitoring Loop"]
        TIMER["Re-Score Timer<br/>(every N seconds)"]
        EVT["Event Triggers"]
        FED["On-Device Federated<br/>Update Interface<br/>(Privacy-Preserving)"]
    end

    %% Data flow
    CAM --> FD
    CAM --> RPPG
    CAM --> PAD
    KBD --> KD
    MOUSE --> MM
    CTX --> DF
    CTX --> GV

    CS -->|"facial_score"| DW
    LF -->|"liveness_score"| DW
    BM -->|"behavioral_score"| DW

    TS --> SM
    TS --> SHAP

    ATTR --> DASHBOARD
    SM --> DASHBOARD
    LOW --> DASHBOARD
    MED --> DASHBOARD
    HIGH --> DASHBOARD

    CONTINUOUS --> SENSING

    style SENSING fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style ENCODERS fill:#0f3460,stroke:#16213e,color:#e0e0e0
    style L1 fill:#1a1a4e,stroke:#533483,color:#e0e0e0
    style L2 fill:#1a1a4e,stroke:#533483,color:#e0e0e0
    style L3 fill:#1a1a4e,stroke:#533483,color:#e0e0e0
    style FUSION fill:#533483,stroke:#e94560,color:#e0e0e0
    style XAI fill:#2d6a4f,stroke:#40916c,color:#e0e0e0
    style POLICY fill:#e94560,stroke:#ff6b6b,color:#e0e0e0
    style RESPONSES fill:#c9184a,stroke:#ff6b6b,color:#e0e0e0
    style DASHBOARD fill:#0077b6,stroke:#00b4d8,color:#e0e0e0
    style CONTINUOUS fill:#6930c3,stroke:#7400b8,color:#e0e0e0
```

## Data Flow Summary

```
Webcam Frame ─────────┬──→ Face Detect → Align → Embed → Cosine Sim ─────→ facial_score
                      ├──→ rPPG Extraction ──────────────┐
                      └──→ PAD CNN ──────────────────────┤→ Liveness Fusion → liveness_score
                                                         │
Keyboard Events ──→ Keystroke Feature Extraction ────────┐
Mouse Events ─────→ Mouse Feature Extraction ────────────┤
Device/IP/Time ───→ Context Features ────────────────────┤→ Sequence Model → behavioral_score
                                                         │
                         ┌───────────────────────────────┘
                         ▼
              ┌─────────────────────┐
              │   Trust Fusion      │       ┌──────────────┐
              │   Engine            │──────→│ SHAP         │──→ Per-Feature Attributions
              │   (Dynamic Weights) │       │ Explainer    │       ↓
              │   → Trust Score     │       └──────────────┘   Dashboard
              └────────┬────────────┘
                       │
                       ▼
              ┌─────────────────────┐
              │  Policy Engine      │
              │  (State Machine)    │
              ├─────────────────────┤
              │ ≥0.80 → Allow       │
              │ 0.50–0.79 → Step-Up │
              │ <0.50 → Kill + Alert│
              └─────────────────────┘
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
         Allow     WebAuthn    SOC Alert
                   Challenge   + Session
                               Termination
```

## Component Boundaries

Each component is designed with clean module boundaries so they can be independently tested, swapped, or upgraded:

| Component | Module Path | Input | Output |
|-----------|-------------|-------|--------|
| Face Detection & Alignment | `ml/facial/alignment.py` | Raw frame (BGR) | Aligned face crop (112×112) |
| Facial Embedding | `ml/facial/embedding.py` | Aligned face crop | 128-dim float vector |
| Face Matching | `ml/facial/matching.py` | Embedding + enrolled template | Cosine similarity score (0–1) |
| rPPG Extraction | `ml/liveness/rppg.py` | Sequence of frames + face ROIs | Pulse signal + confidence |
| Presentation Attack Detector | `ml/liveness/pad.py` | Single frame (face crop) | Spoof probability (0–1) |
| Liveness Fusion | `ml/liveness/fusion.py` | rPPG confidence + PAD probability | Liveness sub-score (0–1) |
| Keystroke Feature Extractor | `ml/behavioral/keystroke.py` | Raw keystroke event stream | Feature vector |
| Mouse Feature Extractor | `ml/behavioral/mouse.py` | Raw mouse event stream | Feature vector |
| Context Feature Extractor | `ml/behavioral/context.py` | Device/IP/time metadata | Feature vector |
| Behavioral Sequence Model | `ml/behavioral/model.py` | Combined feature vectors over time | Behavioral consistency score (0–1) |
| Trust Fusion Engine | `ml/fusion/engine.py` | 3 layer scores + confidences | Trust Score (0–1) + weights used |
| SHAP Explainer | `ml/explainability/shap_explainer.py` | Trust fusion inputs/outputs | Per-feature attributions |
| Policy Engine | `backend/app/services/policy_engine.py` | Trust Score | Risk tier + action |
| Session Manager | `backend/app/services/session.py` | Session events | Session state |
| WebAuthn Step-Up | `backend/app/services/webauthn.py` | Step-up trigger | Challenge/response flow |

## Database Schema (SQLite Prototype)

```sql
-- User enrollment
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    facial_template BLOB,          -- Enrolled 128-dim embedding
    behavioral_baseline BLOB,      -- Serialized baseline model
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Active sessions
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(id),
    trust_score REAL DEFAULT 1.0,
    risk_tier TEXT DEFAULT 'low',   -- low | medium | high
    status TEXT DEFAULT 'active',   -- active | step_up | terminated
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trust score history (for dashboard timeline)
CREATE TABLE trust_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    trust_score REAL NOT NULL,
    facial_score REAL,
    liveness_score REAL,
    behavioral_score REAL,
    shap_attributions TEXT,         -- JSON blob
    risk_tier TEXT,
    policy_action TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- SOC alerts
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    alert_type TEXT NOT NULL,       -- step_up_required | session_terminated
    details TEXT,                   -- JSON blob with context
    acknowledged BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- WebAuthn credentials (for step-up)
CREATE TABLE webauthn_credentials (
    id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(id),
    credential_id BLOB NOT NULL,
    public_key BLOB NOT NULL,
    sign_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Security-Relevant Simplifications (Prototype)

> [!CAUTION]
> The following are deliberate prototype simplifications that must be addressed before any production deployment:

1. **Anti-spoofing model**: Stubbed with a placeholder CNN — not trained on a real presentation-attack dataset
2. **rPPG pipeline**: Uses a well-established algorithm but has not been validated against sophisticated attacks
3. **WebAuthn**: Running in demo/attestation-none mode — no hardware-backed attestation verification
4. **SQLite**: Not suitable for concurrent production workloads — swap for PostgreSQL
5. **Federated learning**: Interface-only stub — no actual on-device training implementation
6. **Session tokens**: Simplified JWT — not using hardware-bound session keys
7. **TLS**: Not enforced in development — all production traffic must use HTTPS
