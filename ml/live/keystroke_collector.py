"""
AdaptiAuth — Live Keystroke Collector
======================================

Captures physical keystroke timing for the target password phrase.
Produces the feature vector (Hold, DD, UD times) used by the behavioral model.

IMPORTANT DESIGN DECISION:
  Feature computation happens on Enter KEY-UP (release), not key-down.
  This is because the Enter key's hold time is part of the feature vector,
  and the hold time can only be measured after the release event arrives.
"""

import time
import threading
from pynput import keyboard


class KeystrokeCollector:
    """
    Listens for a specific sequence of keystrokes and computes
    the timing feature vector: Hold times, Down-Down (DD), Up-Down (UD).

    For target_sequence="start\\n" (6 keys), produces:
        6 Hold + 5 DD + 5 UD = 16 features.
    """
    def __init__(self, target_sequence="start\n"):
        self.target_sequence = target_sequence
        self.key_events = []  # list of {'key': str, 'action': 'press'/'release', 'time': float}
        self.pressed_keys = set()
        self.listener = None
        self.lock = threading.Lock()

        self.current_typed = ""
        self.last_feature_vector = None  # None = not ready, list = valid, "INVALID_PHRASE" = wrong
        self._enter_pressed = False  # flag: Enter key-down seen, waiting for key-up

    def start(self):
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def stop(self):
        if self.listener:
            self.listener.stop()

    def reset(self):
        with self.lock:
            self.key_events.clear()
            self.current_typed = ""
            self.last_feature_vector = None
            self._enter_pressed = False
            self.pressed_keys.clear()

    def _key_to_str(self, key):
        """Convert a pynput key object to a normalized string."""
        try:
            c = key.char
            if c is None:
                return None
            if c == '\r':
                return '\n'
            return c
        except AttributeError:
            if key == keyboard.Key.enter:
                return "\n"
            elif key == keyboard.Key.space:
                return " "
            elif key == keyboard.Key.shift or key == keyboard.Key.shift_r:
                return "Shift"
            elif key == keyboard.Key.backspace:
                return "Backspace"
            return None  # Ignore all other special keys (Ctrl, Alt, etc.)

    def on_press(self, key):
        char = self._key_to_str(key)
        if char is None:
            return

        with self.lock:
            # Backspace resets the entire buffer
            if char == "Backspace":
                self.key_events.clear()
                self.current_typed = ""
                self._enter_pressed = False
                return

            # Ignore Shift (used for capitalization, not a character)
            if char == "Shift":
                return

            # Ignore auto-repeat holds
            if char in self.pressed_keys:
                return

            self.pressed_keys.add(char)
            self.key_events.append({'key': char, 'action': 'press', 'time': time.time()})

            if char == "\n":
                # Mark that Enter was pressed — we'll compute on release
                self._enter_pressed = True
            elif len(char) == 1:
                self.current_typed += char

    def on_release(self, key):
        char = self._key_to_str(key)
        if char is None:
            return

        with self.lock:
            if char in self.pressed_keys:
                self.pressed_keys.remove(char)

            if char == "Shift":
                return

            self.key_events.append({'key': char, 'action': 'release', 'time': time.time()})

            # Compute features ONLY on Enter RELEASE
            # This ensures the Enter key's hold time is captured
            if char == "\n" and self._enter_pressed:
                self._enter_pressed = False
                self._compute_features()

    def _compute_features(self):
        """
        Match press/release pairs, validate the typed sequence, and extract
        the timing feature vector.
        """
        presses = []
        releases_pool = []

        for ev in self.key_events:
            if ev['action'] == 'press':
                presses.append(ev)
            else:
                releases_pool.append(ev)

        # Match each press to its earliest corresponding release
        releases = list(releases_pool)  # working copy
        matched = []
        for p in presses:
            for r in releases:
                if r['key'] == p['key'] and r['time'] >= p['time']:
                    matched.append((p['time'], r['time'], p['key']))
                    releases.remove(r)
                    break

        matched.sort(key=lambda x: x[0])

        # We need exactly N matched pairs for the target sequence
        n = len(self.target_sequence)
        if len(matched) >= n:
            seq = matched[-n:]

            # Validate character sequence
            target_chars = list(self.target_sequence)
            seq_chars = [x[2] for x in seq]

            if seq_chars == target_chars:
                features = []
                for i in range(n):
                    pt_i, rt_i, _ = seq[i]

                    # Hold time H.i
                    h = rt_i - pt_i
                    features.append(h)

                    if i < n - 1:
                        pt_next, rt_next, _ = seq[i + 1]
                        # Down-Down time
                        dd = pt_next - pt_i
                        features.append(dd)
                        # Up-Down time
                        ud = pt_next - rt_i
                        features.append(ud)

                self.last_feature_vector = features
            else:
                self.last_feature_vector = "INVALID_PHRASE"
        else:
            self.last_feature_vector = None

        # Always clear buffer for next attempt
        self.current_typed = ""
        self.key_events.clear()

    def get_last_feature_vector(self):
        """
        Thread-safe retrieval. Returns:
          - list of floats: valid feature vector
          - "INVALID_PHRASE": wrong characters typed
          - None: not ready yet
        Consumes the value (resets to None after reading).
        """
        with self.lock:
            vec = self.last_feature_vector
            self.last_feature_vector = None
            return vec
