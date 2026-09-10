import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

export default function LoginDemo() {
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [trustScore, setTrustScore] = useState(null);
  const [behaviorCount, setBehaviorCount] = useState({ key: 0, mouse: 0 });
  const navigate = useNavigate();

  useEffect(() => {
    // Start webcam
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(s => {
        setStream(s);
        if (videoRef.current) {
          videoRef.current.srcObject = s;
        }
      })
      .catch(err => console.error("Webcam error:", err));

    return () => {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  useEffect(() => {
    // Behavior tracking mock
    const handleKey = () => setBehaviorCount(prev => ({ ...prev, key: prev.key + 1 }));
    const handleMouse = () => setBehaviorCount(prev => ({ ...prev, mouse: prev.mouse + 1 }));

    window.addEventListener('keydown', handleKey);
    window.addEventListener('mousemove', handleMouse);

    return () => {
      window.removeEventListener('keydown', handleKey);
      window.removeEventListener('mousemove', handleMouse);
    };
  }, []);

  useEffect(() => {
    // Poll trust score
    const interval = setInterval(() => {
      fetch('/api/trust/demo-session/rescore', { method: 'POST' })
        .then(r => r.json())
        .then(data => setTrustScore(data))
        .catch(() => {});
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="container" style={{ padding: '2rem 0' }}>
      <h1>Demo Login & Session</h1>
      <p>Simulating continuous biometric capture.</p>
      
      <div style={{ display: 'flex', gap: '2rem', marginTop: '2rem' }}>
        <div className="card" style={{ flex: 1 }}>
          <h2>Webcam (Layer 1 & 2)</h2>
          <div style={{ background: '#000', borderRadius: '8px', overflow: 'hidden' }}>
            <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', display: 'block' }} />
          </div>
        </div>

        <div className="card" style={{ flex: 1 }}>
          <h2>Live Trust Score</h2>
          {trustScore ? (
            <div>
              <h1 style={{ fontSize: '4rem', color: trustScore.risk_tier === 'low' ? '#4ade80' : trustScore.risk_tier === 'medium' ? '#facc15' : '#ef4444' }}>
                {(trustScore.trust_score * 100).toFixed(0)}%
              </h1>
              <p>Risk Tier: <strong style={{ textTransform: 'uppercase' }}>{trustScore.risk_tier}</strong></p>
              <p>Action: <strong>{trustScore.policy_action}</strong></p>
              <hr style={{ margin: '1rem 0', borderColor: '#333' }} />
              <p>Facial: {trustScore.facial_score?.toFixed(2)}</p>
              <p>Liveness: {trustScore.liveness_score?.toFixed(2)}</p>
              <p>Behavioral: {trustScore.behavioral_score?.toFixed(2)}</p>
            </div>
          ) : (
            <p>Loading score...</p>
          )}

          <div style={{ marginTop: '2rem', background: '#222', padding: '1rem', borderRadius: '8px' }}>
            <h3>Telemetry (Layer 3)</h3>
            <p>Keystrokes: {behaviorCount.key}</p>
            <p>Mouse Movements: {behaviorCount.mouse}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
