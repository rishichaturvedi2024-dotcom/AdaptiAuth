import React, { useState, useEffect } from 'react';

export default function SOCDashboard() {
  const [explanation, setExplanation] = useState(null);

  useEffect(() => {
    fetch('/api/trust/demo-session/explain')
      .then(r => r.json())
      .then(data => setExplanation(data))
      .catch(() => {});
  }, []);

  return (
    <div className="container" style={{ padding: '2rem 0' }}>
      <h1>SOC Audit Dashboard</h1>
      <p>Continuous monitoring and explainability view.</p>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '2rem' }}>
        <div className="card">
          <h2>Active Sessions</h2>
          <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #333' }}>
                <th style={{ padding: '0.5rem 0' }}>Session ID</th>
                <th>Trust Score</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ padding: '0.5rem 0' }}>demo-session</td>
                <td>{explanation ? (explanation.trust_score * 100).toFixed(0) + '%' : '...'}</td>
                <td><span className={`trust-badge trust-badge--${explanation?.risk_tier || 'low'}`}>{explanation?.risk_tier || 'low'}</span></td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="card">
          <h2>Trust Score Explainability (SHAP)</h2>
          {explanation ? (
            <div>
              <p>Base Value: {explanation.base_value}</p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '1rem' }}>
                {explanation.attributions.map((attr, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#222', padding: '0.5rem', borderRadius: '4px' }}>
                    <span>{attr.feature}</span>
                    <span style={{ color: attr.direction === 'positive' ? '#4ade80' : '#ef4444' }}>
                      {attr.direction === 'positive' ? '+' : '-'}{attr.attribution.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p>Loading explanation...</p>
          )}
        </div>
      </div>
    </div>
  );
}
