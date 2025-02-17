import cv2
import mediapipe as mp
import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Constants
WIDTH, HEIGHT = 800, 600
GRAVITY = 9.81  # m/s^2
FPS = 60
DRAG_COEFFICIENT = 0.01  # Air resistance coefficient
BALL_RADIUS = 10
OBSTACLE_RADIUS = 30

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Classes
class Ball:
    def __init__(self, x, y, velocity, angle):
        self.x = x
        self.y = y
        self.vx = velocity * np.cos(np.radians(angle))  # Horizontal velocity
        self.vy = -velocity * np.sin(np.radians(angle))  # Vertical velocity
        self.radius = BALL_RADIUS
        self.color = (255, 0, 0)
        self.trail = []

    def update(self, dt):
        # Apply gravity and drag
        self.vx -= DRAG_COEFFICIENT * self.vx * dt
        self.vy += GRAVITY * dt - DRAG_COEFFICIENT * self.vy * dt

        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Store trail points
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 50:  # Limit trail length
            self.trail.pop(0)

    def draw(self, screen):
        # Draw ball
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

        # Draw trail
        if len(self.trail) > 1:
            pygame.draw.lines(screen, (255, 165, 0), False, self.trail, 2)

    def check_collision(self, obstacles):
        for obstacle in obstacles:
            distance = np.sqrt((self.x - obstacle.x)**2 + (self.y - obstacle.y)**2)
            if distance < self.radius + obstacle.radius:
                return True
        return False


class Obstacle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = OBSTACLE_RADIUS
        self.color = (0, 0, 255)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


def detect_hand(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract index finger tip and wrist positions
        index_finger_tip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        wrist = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]

        # Calculate angle and velocity magnitude
        dx = index_finger_tip.x - wrist.x
        dy = index_finger_tip.y - wrist.y
        angle = np.degrees(np.arctan2(dy, dx))
        velocity = np.sqrt(dx**2 + dy**2) * 100  # Scale for better control

        return angle, velocity
    return None, None


def plot_energy(ke, pe):
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(ke, label="Kinetic Energy", color="blue")
    ax.plot(pe, label="Potential Energy", color="green")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy (J)")
    ax.legend()
    canvas = FigureCanvas(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    return pygame.image.frombuffer(raw_data, size, "RGB")


def main():
    cap = cv2.VideoCapture(0)
    ball = None
    obstacles = [Obstacle(400, 300)]  # Initial obstacle
    ke_history = []
    pe_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        angle, velocity = detect_hand(frame)

        # Launch a new ball if hand gesture is detected
        if angle is not None and velocity is not None:
            ball = Ball(50, HEIGHT - 50, velocity, angle)

        # Physics simulation
        if ball:
            dt = clock.tick(FPS) / 1000  # Time step in seconds
            ball.update(dt)

            # Check collision
            if ball.check_collision(obstacles):
                print("Collision detected!")
                ball = None

            # Calculate energy
            ke = 0.5 * (ball.vx**2 + ball.vy**2)  # Kinetic energy
            pe = GRAVITY * (HEIGHT - ball.y)  # Potential energy
            ke_history.append(ke)
            pe_history.append(pe)

        # Render Pygame scene
        screen.fill((255, 255, 255))
        if ball:
            ball.draw(screen)
        for obstacle in obstacles:
            obstacle.draw(screen)

        # Plot energy graph
        if ke_history and pe_history:
            energy_plot = plot_energy(ke_history, pe_history)
            screen.blit(energy_plot, (WIDTH - 400, 10))

        pygame.display.flip()

        # Display webcam feed
        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
