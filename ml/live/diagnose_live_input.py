"""
AdaptiAuth — Live Keyboard Diagnostic Tool
============================================

This script tests the REAL physical keyboard capture independently from the
full ML pipeline. It proves whether pynput correctly receives all characters
of the password 'start' + Enter, and whether the feature extractor produces
the correct 16-dimensional vector.

Usage:
    backend\\venv\\Scripts\\python.exe ml/live/diagnose_live_input.py

Instructions:
    1. Run this script.
    2. Physically type: start
    3. Physically press: Enter
    4. The script will print every key event and the extracted feature vector.
    5. Press Escape to quit.
"""

import sys
import os
import time
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pynput import keyboard


class DiagnosticCollector:
    """
    Diagnostic version of KeystrokeCollector that logs everything
    and computes features AFTER the Enter key-up (not key-down).
    """
    def __init__(self, target_phrase="start"):
        self.target_phrase = target_phrase
        self.key_events = []
        self.pressed_keys = set()
        self.current_typed = ""
        self.lock = threading.Lock()
        self.enter_pressed = False

    def on_press(self, key):
        t = time.time()
        char = self._key_to_str(key)
        print(f"  [PRESS]   t={t:.4f}  raw={key!r:30s}  type={type(key).__name__:15s}  char={char!r}")

        if char is None:
            return
        if key == keyboard.Key.esc:
            return False

        with self.lock:
            if char == "Backspace":
                self.key_events.clear()
                self.current_typed = ""
                print("  >>> BACKSPACE: buffer cleared")
                return

            if char in self.pressed_keys:
                print(f"  >>> AUTO-REPEAT ignored for {char!r}")
                return

            self.pressed_keys.add(char)
            self.key_events.append({'key': char, 'action': 'press', 'time': t})

            if char == "\n":
                self.enter_pressed = True
                print(f"  >>> ENTER key-DOWN recorded. Will compute on key-UP.")
            elif len(char) == 1:
                self.current_typed += char
                print(f"  >>> current_typed = {self.current_typed!r}")

    def on_release(self, key):
        t = time.time()
        char = self._key_to_str(key)
        print(f"  [RELEASE] t={t:.4f}  raw={key!r:30s}  type={type(key).__name__:15s}  char={char!r}")

        if char is None:
            return

        with self.lock:
            if char in self.pressed_keys:
                self.pressed_keys.remove(char)

            self.key_events.append({'key': char, 'action': 'release', 'time': t})

            # Compute features on Enter RELEASE so the Enter hold time is captured
            if char == "\n" and self.enter_pressed:
                self.enter_pressed = False
                self._compute_and_report()

    def _key_to_str(self, key):
        try:
            c = key.char
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
            elif key == keyboard.Key.esc:
                return "Esc"
            return str(key)

    def _compute_and_report(self):
        print("\n" + "="*60)
        print("COMPUTING FEATURES")
        print("="*60)

        presses = [ev for ev in self.key_events if ev['action'] == 'press']
        releases_pool = [ev for ev in self.key_events if ev['action'] == 'release']

        print(f"Total press events: {len(presses)}")
        print(f"Total release events: {len(releases_pool)}")

        for i, p in enumerate(presses):
            print(f"  Press {i}: key={p['key']!r} t={p['time']:.4f}")
        for i, r in enumerate(releases_pool):
            print(f"  Release {i}: key={r['key']!r} t={r['time']:.4f}")

        # Match presses to releases
        releases = list(releases_pool)  # copy so we can remove
        matched = []
        for p in presses:
            for r in releases:
                if r['key'] == p['key'] and r['time'] >= p['time']:
                    matched.append((p['time'], r['time'], p['key']))
                    releases.remove(r)
                    break

        matched.sort(key=lambda x: x[0])
        print(f"\nMatched press-release pairs: {len(matched)}")
        for i, (pt, rt, k) in enumerate(matched):
            print(f"  [{i}] key={k!r}  press={pt:.4f}  release={rt:.4f}  hold={rt-pt:.4f}s")

        # Target sequence including Enter
        target_with_enter = self.target_phrase + "\n"
        n = len(target_with_enter)
        print(f"\nTarget sequence: {list(target_with_enter)}")
        print(f"Required matched pairs: {n}")

        if len(matched) < n:
            print(f"\nFAILED: Only {len(matched)} matched pairs, need {n}")
            self.key_events.clear()
            self.current_typed = ""
            return

        seq = matched[-n:]
        seq_chars = [x[2] for x in seq]
        target_chars = list(target_with_enter)

        print(f"Last {n} matched chars: {seq_chars}")
        print(f"Expected chars:         {target_chars}")

        if seq_chars != target_chars:
            print("\nFAILED: Character sequence mismatch!")
            self.key_events.clear()
            self.current_typed = ""
            return

        # Extract features: H, DD, UD
        features = []
        for i in range(n):
            pt_i, rt_i, _ = seq[i]
            h = rt_i - pt_i
            features.append(h)
            if i < n - 1:
                pt_next, rt_next, _ = seq[i + 1]
                dd = pt_next - pt_i
                features.append(dd)
                ud = pt_next - rt_i
                features.append(ud)

        print(f"\nSUCCESS! Feature vector extracted.")
        print(f"Dimensionality: {len(features)}")
        print(f"Expected: {n} holds + {n-1} DDs + {n-1} UDs = {n + 2*(n-1)}")
        print(f"Features: {[round(f, 4) for f in features]}")

        self.key_events.clear()
        self.current_typed = ""
        print("\nBuffer cleared. Type again to test, or press Escape to quit.")


def main():
    print("="*60)
    print("AdaptiAuth — Live Keyboard Diagnostic")
    print("="*60)
    print()
    print(f"Target phrase: 'start' + Enter")
    print(f"Expected sequence: ['s', 't', 'a', 'r', 't', '\\n']")
    print(f"Expected feature dimensionality: 6 holds + 5 DDs + 5 UDs = 16")
    print()
    print("Instructions:")
    print("  1. Type: start")
    print("  2. Press: Enter")
    print("  3. Watch the output")
    print("  4. Press Escape to quit")
    print()
    print("Listening for physical keyboard events...")
    print("-"*60)

    collector = DiagnosticCollector(target_phrase="start")
    listener = keyboard.Listener(
        on_press=collector.on_press,
        on_release=collector.on_release
    )
    listener.start()
    listener.join()
    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
