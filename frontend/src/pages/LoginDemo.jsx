import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

export default function LoginDemo() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const eventsRef = useRef({ keystrokes: [], mouse_events: [] });
  const [stream, setStream] = useState(null);
  const [trustScore, setTrustScore] = useState(null);
  const [behaviorCount, setBehaviorCount] = useState({ key: 0, mouse: 0 });
  const [faceDetected, setFaceDetected] = useState(null);
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
      .catch(err => {
        console.error("Webcam error:", err);
        setFaceDetected(false);
      });

    return () => {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  useEffect(() => {
    // Behavior tracking mock
    const handleKey = (event) => {
      eventsRef.current.keystrokes.push({
        key: event.key,
        event_type: 'keydown',
        timestamp: Date.now(),
      });
      setBehaviorCount(prev => ({ ...prev, key: prev.key + 1 }));
    };
    const handleMouse = (event) => {
      eventsRef.current.mouse_events.push({
        x: event.clientX,
        y: event.clientY,
        event_type: 'mousemove',
        timestamp: Date.now(),
      });
      setBehaviorCount(prev => ({ ...prev, mouse: prev.mouse + 1 }));
    };

    window.addEventListener('keydown', handleKey);
    window.addEventListener('mousemove', handleMouse);

    return () => {
      window.removeEventListener('keydown', handleKey);
      window.removeEventListener('mousemove', handleMouse);
    };
  }, []);

  useEffect(() => {
    const sendBehavior = async () => {
      const captured = eventsRef.current;
      eventsRef.current = { keystrokes: [], mouse_events: [] };
      if (!captured.keystrokes.length && !captured.mouse_events.length) return;
      await fetch('/api/biometrics/behavior', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: 'demo-session',
          ...captured,
          timestamp: Date.now(),
          device_fingerprint: navigator.userAgent,
        }),
      });
    };

    const sendFrame = async () => {
      if (!videoRef.current || videoRef.current.readyState < 2) {
        setFaceDetected(false);
        await fetch('/api/biometrics/liveness/absent', {
          method: 'POST',
          body: new URLSearchParams({ session_id: 'demo-session' }),
        });
        return;
      }
      const canvas = canvasRef.current;
      const video = videoRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.75));
      if (!blob) return;
      const form = new FormData();
      form.append('file', blob, 'frame.jpg');
      form.append('session_id', 'demo-session');
      const response = await fetch('/api/biometrics/liveness', { method: 'POST', body: form });
      if (response.ok) {
        const result = await response.json();
        setFaceDetected(result.face_detected);
      }
    };

    const updateTrust = async () => {
      const response = await fetch('/api/trust/demo-session/rescore', { method: 'POST' });
      if (response.ok) setTrustScore(await response.json());
    };

    const behaviorTimer = setInterval(() => sendBehavior().catch(() => {}), 2000);
    const frameTimer = setInterval(() => sendFrame().catch(() => {}), 3000);
    const trustTimer = setInterval(() => updateTrust().catch(() => {}), 3000);
    updateTrust().catch(() => {});
    return () => {
      clearInterval(behaviorTimer);
      clearInterval(frameTimer);
      clearInterval(trustTimer);
    };
  }, []);

  return (
    <div className="container" style={{ padding: '2rem 0' }}>
      <h1>Demo Login & Session</h1>
      <p>Simulating continuous biometric capture.</p>
      {faceDetected === false && (
        <p style={{ color: '#ef4444', fontWeight: 'bold' }}>
      No face detected — trust score reduced.
        </p>
      )}
      
      <div style={{ display: 'flex', gap: '2rem', marginTop: '2rem' }}>
        <div className="card" style={{ flex: 1 }}>
          <h2>Webcam (Layer 1 & 2)</h2>
          <div style={{ background: '#000', borderRadius: '8px', overflow: 'hidden' }}>
            <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', display: 'block' }} />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
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
