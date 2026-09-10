"""
AdaptiAuth — ML Layer 2: Remote Photoplethysmography (rPPG)
Extracts a pulse signal from a sequence of frames.
Phase 10 Upgraded: DSP pipeline (Bandpass + FFT SNR).

CALIBRATION NOTE (Phase 10 Live Audit):
  The ROI must be the detected face region, not the center of the full frame.
  Using the full frame center-crop dilutes the physiological signal with
  background scene noise, crushing the SNR.
"""

import numpy as np
from scipy.signal import butter, filtfilt
import scipy.fft as fft

class RPPGExtractor:
    """
    Extracts physiological liveness confidence based on rPPG.
    """
    def __init__(self, fps=30):
        self.fps = fps
        self.max_frames = fps * 5  # Up to 5 seconds of buffer
        self.min_frames = fps * 1  # Need at least 1s to compute FFT
        self.buffer = []

    def process_frame(self, frame: np.ndarray, face_box=None) -> float:
        """
        Process a single frame and update the buffer.
        Returns a confidence score based on the SNR of the pulse signal.
        
        Args:
            frame: Full BGR webcam frame (H, W, 3)
            face_box: Optional (x1, y1, x2, y2) bounding box of the detected face.
                      When provided, the green channel is extracted from the 
                      forehead/cheek region (upper 60% of the face box).
                      When None, falls back to center-crop of full frame.
        """
        h, w, _ = frame.shape
        
        if face_box is not None:
            # Use face bounding box — extract forehead/cheek region (upper 60%)
            x1, y1, x2, y2 = face_box
            # Clamp to frame bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            
            face_h = y2 - y1
            if face_h <= 0 or x2 - x1 <= 0:
                return 0.0
                
            # Upper 60% of face box = forehead + cheeks (best rPPG signal)
            # Avoid eyes/mouth which have more movement artifacts
            roi_y2 = y1 + int(face_h * 0.6)
            # Also narrow horizontally to avoid ears/hair — inner 60%
            face_w = x2 - x1
            roi_x1 = x1 + int(face_w * 0.2)
            roi_x2 = x2 - int(face_w * 0.2)
            
            roi = frame[y1:roi_y2, roi_x1:roi_x2]
        else:
            # Fallback: center crop of full frame (original behavior)
            roi = frame[h//4:3*h//4, w//4:3*w//4]
        
        if roi.size == 0:
            return 0.0
            
        green_mean = np.mean(roi[:, :, 1])
        self.buffer.append(green_mean)
        
        if len(self.buffer) > self.max_frames:
            self.buffer.pop(0)
            
        if len(self.buffer) < self.min_frames:
            return 0.5 # Unknown/neutral until buffer fills
            
        # 2. Detrend the signal
        signal = np.array(self.buffer)
        signal = signal - np.mean(signal)
        
        # 3. Butterworth Bandpass Filter (0.75 - 4.0 Hz) -> 45 to 240 BPM
        # Protect against signals that are too short for filtfilt's default padding
        if len(signal) <= 9: # padlen is usually 3 * max(len(a), len(b))
            return 0.5
            
        nyq = 0.5 * self.fps
        low = 0.75 / nyq
        high = 4.0 / nyq
        b, a = butter(3, [low, high], btype='band')
        try:
            filtered_signal = filtfilt(b, a, signal)
        except ValueError:
            return 0.5
            
        # 4. FFT to find the power spectral density
        N = len(filtered_signal)
        # Hann window to reduce leakage
        windowed = filtered_signal * np.hanning(N)
        yf = fft.rfft(windowed)
        xf = fft.rfftfreq(N, 1.0 / self.fps)
        
        power = np.abs(yf)**2
        
        # Filter power spectrum to our target heart rate range
        valid_idx = np.where((xf >= 0.75) & (xf <= 4.0))[0]
        if len(valid_idx) == 0:
            return 0.0
            
        # 5. Calculate SNR
        # SNR = Power of dominant frequency (and its harmonics/neighbors) / Power of other frequencies
        dominant_idx = valid_idx[np.argmax(power[valid_idx])]
        
        # Power around the dominant frequency (e.g. +/- 1 bin)
        signal_power = 0
        for i in range(max(0, dominant_idx-1), min(len(power), dominant_idx+2)):
            signal_power += power[i]
            
        total_power = np.sum(power[valid_idx])
        noise_power = total_power - signal_power
        
        # Avoid division by zero
        if noise_power <= 0:
            snr = 10.0 # Arbitrary high max value
        else:
            snr = signal_power / noise_power
            
        # 6. Map SNR to confidence score
        # Typically, SNR > 2 or 3 is a strong indicator of a real physiological pulse.
        # We will map SNR=[0.5, 4.0] to Confidence=[0.0, 1.0]
        mapped_confidence = (snr - 0.5) / 3.5
        
        return min(max(mapped_confidence, 0.0), 1.0)

