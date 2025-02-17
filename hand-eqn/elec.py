import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Particle class
class Particle:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.random.uniform(-1, 1, size=2) * 2
        self.acc = np.zeros(2)
        self.lifespan = random.uniform(50, 100)
        self.size = random.randint(2, 5)

    def update(self, attractors, particles, mode="attract"):
        # Apply attraction/repulsion force from attractors (hand landmarks)
        for attractor in attractors:
            force = attractor - self.pos
            distance = np.linalg.norm(force)
            if distance > 0:
                force = force / distance  # Normalize force
                strength = 0.1 * (1 / distance)  # Inverse square law
                if mode == "repel":
                    force *= -1  # Reverse force for repulsion
                self.acc += force * strength

        # Apply repulsion force between particles
        for other in particles:
            if other != self:
                force = self.pos - other.pos
                distance = np.linalg.norm(force)
                if 0 < distance < 50:  # Only apply repulsion within a certain range
                    force = force / distance
                    strength = 0.5 * (1 / distance)  # Repulsion strength
                    self.acc += force * strength

        # Update velocity and positionc
        self.vel += self.acc
        self.pos += self.vel
        self.acc *= 0.0  # Reset acceleration

        # Bounce off screen edges
        h, w = frame.shape[:2]
        if self.pos[0] <= 0 or self.pos[0] >= w:
            self.vel[0] *= -1
        if self.pos[1] <= 0 or self.pos[1] >= h:
            self.vel[1] *= -1

        self.lifespan -= 1

    def draw(self, canvas):
        if self.lifespan > 0:
            # Change color based on velocity
            speed = np.linalg.norm(self.vel)
            base_color = (
                int(255 * (1 - speed / 10)),  # Red decreases with speed
                int(255 * (speed / 10)),      # Green increases with speed
                255                           # Blue stays constant
            )

            # Add glow effect
            for i in range(3):  # Draw multiple concentric circles
                glow_size = self.size + i * 2
                glow_opacity = 0.8 - i * 0.2
                glow_color = tuple(int(c * glow_opacity) for c in base_color)
                cv2.circle(canvas, (int(self.pos[0]), int(self.pos[1])), glow_size, glow_color, -1)

# Initialize webcam
cap = cv2.VideoCapture(0)

# List to store particles
particles = []

# Initialize trail_canvas dynamically based on the first frame
ret, frame = cap.read()
if not ret:
    raise ValueError("Failed to capture video from webcam.")
trail_canvas = np.zeros_like(frame)

# Create a dynamic background
def create_dynamic_background(shape, mode="stars"):
    background = np.zeros(shape, dtype=np.uint8)
    if mode == "stars":
        num_stars = 200
        for _ in range(num_stars):
            x = random.randint(0, shape[1] - 1)
            y = random.randint(0, shape[0] - 1)
            intensity = random.randint(150, 255)
            cv2.circle(background, (x, y), 1, (intensity, intensity, intensity), -1)
    elif mode == "grid":
        for x in range(0, shape[1], 20):
            cv2.line(background, (x, 0), (x, shape[0]), (50, 50, 255), 1)
        for y in range(0, shape[0], 20):
            cv2.line(background, (0, y), (shape[1], y), (50, 50, 255), 1)
    return background

dynamic_background = create_dynamic_background(frame.shape, mode="stars")

# Gesture detection variables
prev_gesture = None
gesture_mode = "attract"  # Default mode

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a more natural feel
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # List to store hand landmarks as attractors
    attractors = []

    # Detect gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions and add them as attractors
            for lm in hand_landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                attractors.append(np.array([cx, cy]))

            # Gesture detection logic
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            thumb_pos = np.array([thumb_tip.x * w, thumb_tip.y * h])
            index_pos = np.array([index_finger_tip.x * w, index_finger_tip.y * h])
            middle_pos = np.array([middle_finger_tip.x * w, middle_finger_tip.y * h])

            # Thumbs up gesture
            if np.linalg.norm(thumb_pos - index_pos) > 100 and np.linalg.norm(thumb_pos - middle_pos) > 100:
                gesture_mode = "freeze"
            else:
                gesture_mode = "attract"

    # Add new particles randomly near the attractors
    if attractors:
        for _ in range(5):  # Add 5 particles per frame
            attractor = random.choice(attractors)
            particles.append(Particle(attractor[0] + random.uniform(-20, 20),
                                      attractor[1] + random.uniform(-20, 20)))

    # Update and draw particles
    for particle in particles[:]:
        particle.update(attractors, particles, mode=gesture_mode)
        particle.draw(trail_canvas)  # Draw particles on the trail canvas

        # Remove dead particles
        if particle.lifespan <= 0:
            particles.remove(particle)

    # Fade out the trail canvas to create a trailing effect
    trail_canvas = cv2.addWeighted(trail_canvas, 0.95, np.zeros_like(trail_canvas), 0.05, 0)

    # Dynamically change the background based on the number of particles
    if len(particles) > 100:
        dynamic_background = create_dynamic_background(frame.shape, mode="grid")
    else:
        dynamic_background = create_dynamic_background(frame.shape, mode="stars")

    # Combine the dynamic background, trail canvas, and original frame
    combined_frame = cv2.addWeighted(frame, 1, trail_canvas, 0.7, 0)
    combined_frame = cv2.addWeighted(combined_frame, 0.8, dynamic_background, 0.2, 0)

    # Display the frame
    cv2.imshow('Hand Detection with Particles', combined_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
