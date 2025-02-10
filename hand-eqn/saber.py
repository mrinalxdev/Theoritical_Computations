import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import threading
import math
import time

class AdvancedHandGestureInstrument:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )

        # Sound Configuration
        self.sample_rate = 44100
        self.instrument_modes = {
            'sine': self.generate_sine_wave,
            'square': self.generate_square_wave,
            'triangle': self.generate_triangle_wave
        }

        # Musical Scale Configurations
        self.scale_modes = {
            'major': [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25],  # C Major
            'minor': [261.63, 293.66, 311.13, 349.23, 392.00, 415.30, 466.16, 523.25],  # C Minor
            'pentatonic': [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25]  # C Pentatonic
        }

        # Instrument State
        self.current_frequency = 0
        self.current_mode = 'sine'
        self.current_scale = 'major'
        self.volume = 0
        self.effect_intensity = 0
        self.stream = None

    def generate_sine_wave(self, frequency, volume):
        """Generate a sine wave."""
        t = np.linspace(0, 1, self.sample_rate, False)
        return volume * np.sin(2 * np.pi * frequency * t)

    def generate_square_wave(self, frequency, volume):
        """Generate a square wave with harmonics."""
        t = np.linspace(0, 1, self.sample_rate, False)
        square_wave = volume * np.sign(np.sin(2 * np.pi * frequency * t))
        return square_wave

    def generate_triangle_wave(self, frequency, volume):
        """Generate a triangle wave."""
        t = np.linspace(0, 1, self.sample_rate, False)
        triangle_wave = volume * (2/np.pi * np.arcsin(np.sin(2 * np.pi * frequency * t)))
        return triangle_wave

    def apply_effects(self, wave):
        """Apply dynamic effects to the sound wave."""
        if self.effect_intensity > 0:
            # Simple tremolo effect
            tremolo_wave = np.sin(2 * np.pi * 5 * np.linspace(0, 1, len(wave)))
            wave *= (1 + self.effect_intensity * tremolo_wave)
        return wave

    def audio_callback(self, outdata, frames, time, status):
        """Audio stream callback with wave generation."""
        if status:
            print(status)

        # Generate wave based on current mode
        wave = self.instrument_modes[self.current_mode](
            self.current_frequency,
            self.volume
        )

        # Apply dynamic effects
        wave = self.apply_effects(wave)

        # Reshape and fill output data
        outdata[:] = wave[:frames, np.newaxis]

    def start_audio_stream(self):
        """Initialize and start audio output stream."""
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()

    def process_hand_gestures(self):
        """Advanced hand gesture processing with multi-modal interactions."""
        cap = cv2.VideoCapture(0)
        last_mode_switch = time.time()

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            # Reset interaction states
            interaction_detected = False

            if results.multi_hand_landmarks:
                for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Pitch control (first hand)
                    if hand_index == 0:
                        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        vertical_position = 1 - index_tip.y
                        note_index = int(vertical_position * 8)
                        note_index = max(0, min(note_index, 7))

                        # Use current scale
                        self.current_frequency = self.scale_modes[self.current_scale][note_index]
                        interaction_detected = True

                    # Secondary hand for mode switching and effects
                    if hand_index == 1:
                        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                        current_time = time.time()

                        # Mode switching logic
                        if current_time - last_mode_switch > 1:
                            if thumb_tip.y < 0.3:  # Top of screen
                                # Cycle through wave modes
                                modes = list(self.instrument_modes.keys())
                                current_index = modes.index(self.current_mode)
                                self.current_mode = modes[(current_index + 1) % len(modes)]
                                last_mode_switch = current_time

                            if thumb_tip.y > 0.7:  # Bottom of screen
                                # Cycle through scales
                                scales = list(self.scale_modes.keys())
                                current_index = scales.index(self.current_scale)
                                self.current_scale = scales[(current_index + 1) % len(scales)]
                                last_mode_switch = current_time

                        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

                        hand_openness = math.sqrt(
                            (wrist.x - pinky_tip.x)**2 +
                            (wrist.y - pinky_tip.y)**2
                        )

                        self.volume = min(max(1 - hand_openness * 5, 0), 1)
                        self.effect_intensity = min(max(hand_openness * 3, 0), 1)
                        interaction_detected = True

            # On-screen information
            cv2.putText(image,
                f"Mode: {self.current_mode.capitalize()} | Scale: {self.current_scale.capitalize()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Advanced Hand Gesture Virtual Instrument', image)

            # Exit on 'q' key press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        if self.stream:
            self.stream.stop()

    def run(self):
        """Start the advanced virtual instrument."""
        audio_thread = threading.Thread(target=self.start_audio_stream)
        audio_thread.start()

        self.process_hand_gestures()

def main():
    instrument = AdvancedHandGestureInstrument()
    instrument.run()

if __name__ == "__main__":
    main()
