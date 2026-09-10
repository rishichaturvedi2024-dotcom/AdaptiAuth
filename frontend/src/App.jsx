/**
 * AdaptiAuth — Main App Component
 */

import { useState, useEffect } from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import LoginDemo from './pages/LoginDemo';
import SOCDashboard from './pages/SOCDashboard';
import './App.css';

const API_BASE = '/api';

function Home({ health, trustScore }) {
  return (
    <div className="app">
      <div className="bg-grid" />
      <div className="bg-glow bg-glow--1" />
      <div className="bg-glow bg-glow--2" />

      <header className="header">
        <div className="container header__inner">
          <div className="header__brand">
            <span className="header__logo">🛡️</span>
            <span className="header__name">AdaptiAuth</span>
          </div>
          <div className="header__status">
            <span className={`status-dot ${health?.status === 'ok' ? 'status-dot--online' : 'status-dot--offline'}`} />
            <span className="header__status-text">
              {health?.status === 'ok' ? 'Backend Online' : 'Backend Offline'}
            </span>
          </div>
        </div>
      </header>

      <main className="main container">
        <section className="hero animate-fade-in">
          <h1 className="hero__title">
            Risk-Adaptive
            <br />
            <span className="hero__highlight">Continuous Authentication</span>
          </h1>
          <p className="hero__subtitle">
            Multi-modal biometric trust scoring with SHAP explainability
            and zero-trust policy enforcement.
          </p>
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginTop: '2rem' }}>
            <Link to="/login" className="button" style={{ background: 'var(--color-primary)', color: '#fff', padding: '0.75rem 1.5rem', borderRadius: '4px', textDecoration: 'none' }}>Launch Login Demo</Link>
            <Link to="/dashboard" className="button" style={{ background: '#333', color: '#fff', padding: '0.75rem 1.5rem', borderRadius: '4px', textDecoration: 'none' }}>View SOC Dashboard</Link>
          </div>
        </section>

        <div className="grid grid--3 animate-slide-up" style={{ animationDelay: '0.2s' }}>
          <div className="card layer-card">
            <div className="layer-card__icon">👤</div>
            <h3 className="layer-card__title">Layer 1</h3>
            <p className="layer-card__label">Deep Facial Verification</p>
            <p className="layer-card__desc">
              CNN-based 128-dim facial embeddings with 3D face alignment for
              identity verification across pose and lighting.
            </p>
          </div>

          <div className="card layer-card">
            <div className="layer-card__icon">💓</div>
            <h3 className="layer-card__title">Layer 2</h3>
            <p className="layer-card__label">Dual-Layer Liveness</p>
            <p className="layer-card__desc">
              Remote photoplethysmography (rPPG) fused with deep-learning
              presentation-attack detection for genuine liveness confirmation.
            </p>
          </div>

          <div className="card layer-card">
            <div className="layer-card__icon">⌨️</div>
            <h3 className="layer-card__title">Layer 3</h3>
            <p className="layer-card__label">Behavioral Monitoring</p>
            <p className="layer-card__desc">
              Keystroke dynamics, mouse movement, device fingerprint, and
              geo-velocity analysis for session-long behavioral consistency.
            </p>
          </div>
        </div>

        <section className="risk-table animate-slide-up" style={{ animationDelay: '0.6s' }}>
          <div className="card">
            <h2 className="risk-table__title">Zero-Trust Policy Engine</h2>
            <table className="risk-table__table">
              <thead>
                <tr>
                  <th>Trust Score</th>
                  <th>Risk Level</th>
                  <th>Automated Response</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>0.80 – 1.00</td>
                  <td><span className="trust-badge trust-badge--low">Low</span></td>
                  <td>Uninterrupted access</td>
                </tr>
                <tr>
                  <td>0.50 – 0.79</td>
                  <td><span className="trust-badge trust-badge--medium">Medium</span></td>
                  <td>FIDO2 / WebAuthn step-up authentication</td>
                </tr>
                <tr>
                  <td>0.00 – 0.49</td>
                  <td><span className="trust-badge trust-badge--high">High</span></td>
                  <td>Session termination + SOC security alert</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      </main>

      <footer className="footer">
        <div className="container footer__inner">
          <p>AdaptiAuth v0.1.0</p>
          <p className="footer__links">
            <a href="/docs" target="_blank" rel="noopener">API Docs</a>
            {' · '}
            <a href="/redoc" target="_blank" rel="noopener">ReDoc</a>
          </p>
        </div>
      </footer>
    </div>
  );
}

function App() {
  const [health, setHealth] = useState(null);
  const [trustScore, setTrustScore] = useState(null);

  useEffect(() => {
    fetch('/health')
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: 'unreachable' }));

    fetch(`${API_BASE}/trust/demo-session/score`)
      .then((r) => r.json())
      .then(setTrustScore)
      .catch(() => {});
  }, []);

  return (
    <Routes>
      <Route path="/" element={<Home health={health} trustScore={trustScore} />} />
      <Route path="/login" element={<LoginDemo />} />
      <Route path="/dashboard" element={<SOCDashboard />} />
    </Routes>
  );
}

export default App;
