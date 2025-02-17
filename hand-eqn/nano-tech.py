import cv2
import mediapipe as mp
import pygame
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
from enum import Enum
import math

class MoleculeType(Enum):
    NANOBOT = "nanobot"
    SUBSTRATE = "substrate"
    TARGET = "target"
    WASTE = "waste"

@dataclass
class Molecule:
    position: np.ndarray
    velocity: np.ndarray
    type: MoleculeType
    radius: float
    charge: float
    energy: float
    bound_to: Set[int] = None

    def __post_init__(self):
        if self.bound_to is None:
            self.bound_to = set()

class NanoHandSimulation:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        success, img = self.cap.read()
        if not success:
            raise RuntimeError("Failed to initialize camera")

        self.width = int(img.shape[1])
        self.height = int(img.shape[0])

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Hand-Controlled Nanotechnology Simulation")

        # Simulation parameters
        self.temperature = 300.0  # Kelvin
        self.viscosity = 1e-3    # Water viscosity
        self.dt = 0.016          # Time step (60 FPS)

        # Colors
        self.colors = {
            MoleculeType.NANOBOT: (50, 150, 255),
            MoleculeType.SUBSTRATE: (200, 200, 200),
            MoleculeType.TARGET: (255, 50, 50),
            MoleculeType.WASTE: (100, 100, 100)
        }

        # Molecules
        self.molecules: List[Molecule] = []
        self.initialize_molecules()

        # Hand interaction parameters
        self.influence_radius = 50
        self.repulsion_strength = 2.0
        self.attraction_strength = 1.0

        # UI elements
        self.font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()
        self.paused = False

        # Statistics
        self.fps = 0
        self.active_nanobots = 0
        self.total_interactions = 0

    def initialize_molecules(self):
        """Initialize nano-particles in the simulation."""
        # Add nanobots
        for _ in range(200):
            self.molecules.append(Molecule(
                position=np.array([
                    random.uniform(0, self.width),
                    random.uniform(0, self.height)
                ]),
                velocity=np.array([
                    random.uniform(-20, 20),
                    random.uniform(-20, 20)
                ]),
                type=MoleculeType.NANOBOT,
                radius=3.0,
                charge=-1.0,
                energy=1.0
            ))

        # Add substrate molecules
        for _ in range(100):
            self.molecules.append(Molecule(
                position=np.array([
                    random.uniform(0, self.width),
                    random.uniform(0, self.height)
                ]),
                velocity=np.array([
                    random.uniform(-10, 10),
                    random.uniform(-10, 10)
                ]),
                type=MoleculeType.SUBSTRATE,
                radius=2.0,
                charge=0.0,
                energy=0.5
            ))

    def apply_hand_influence(self, hand_landmarks):
        """Apply forces to molecules based on hand position."""
        for landmark in hand_landmarks.landmark:
            hand_pos = np.array([landmark.x * self.width, landmark.y * self.height])

            for molecule in self.molecules:
                diff = hand_pos - molecule.position
                distance = np.linalg.norm(diff)

                if distance < self.influence_radius:
                    # Normalize direction vector
                    direction = diff / (distance + 1e-6)

                    # Calculate force based on distance
                    force_magnitude = (1 - distance/self.influence_radius)

                    # Apply attractive or repulsive force based on finger position
                    if landmark.y < 0.5:  # Upper half of screen - attract
                        force = direction * force_magnitude * self.attraction_strength
                    else:  # Lower half of screen - repel
                        force = -direction * force_magnitude * self.repulsion_strength

                    # Update molecule velocity
                    molecule.velocity += force * self.dt
                    self.total_interactions += 1

    def update_molecules(self):
        """Update all molecules in the simulation."""
        if self.paused:
            return

        for molecule in self.molecules:
            # Apply Brownian motion
            molecule.velocity += np.random.normal(0, 1, 2) * np.sqrt(self.dt * self.temperature)

            # Update position
            molecule.position += molecule.velocity * self.dt

            # Apply damping
            molecule.velocity *= 0.98

            # Boundary conditions (wrap around)
            molecule.position[0] = molecule.position[0] % self.width
            molecule.position[1] = molecule.position[1] % self.height

    def draw_molecules(self, surface):
        """Draw all molecules."""
        for molecule in self.molecules:
            pygame.draw.circle(
                surface,
                self.colors[molecule.type],
                molecule.position.astype(int),
                int(molecule.radius)
            )

    def draw_hand_landmarks(self, surface, hand_landmarks):
        """Draw hand landmarks and their influence areas."""
        for landmark in hand_landmarks.landmark:
            pos = (int(landmark.x * self.width), int(landmark.y * self.height))

            # Draw influence area
            pygame.draw.circle(
                surface,
                (255, 255, 255, 50),  # Semi-transparent white
                pos,
                self.influence_radius,
                1
            )

            # Draw landmark point
            pygame.draw.circle(
                surface,
                (255, 255, 255),
                pos,
                4
            )

    def draw_stats(self, surface):
        """Draw simulation statistics."""
        stats = [
            f"FPS: {self.fps:.1f}",
            f"Temperature: {self.temperature:.1f} K",
            f"Interactions: {self.total_interactions}",
            f"Particles: {len(self.molecules)}",
            "Space: Pause",
            "Up/Down: Temperature"
        ]

        for i, text in enumerate(stats):
            surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (10, 10 + i * 25))

    def run(self):
        """Main simulation loop."""
        running = True

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_UP:
                        self.temperature *= 1.1
                    elif event.key == pygame.K_DOWN:
                        self.temperature *= 0.9

            try:
                # Process camera frame
                success, img = self.cap.read()
                if not success:
                    continue

                # Convert image
                img = cv2.flip(img, 1)
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(imgRGB)

                # Create effect surface
                effect_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

                # Update simulation
                self.update_molecules()

                # Process hand landmarks if detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.apply_hand_influence(hand_landmarks)
                        self.draw_hand_landmarks(effect_surface, hand_landmarks)

                # Draw molecules
                self.draw_molecules(effect_surface)

                # Convert camera frame to Pygame surface
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.rot90(img)
                img_surface = pygame.surfarray.make_surface(img)

                # Combine everything
                self.screen.blit(img_surface, (0, 0))
                self.screen.blit(effect_surface, (0, 0))

                # Draw stats
                self.draw_stats(self.screen)

                # Update display
                pygame.display.flip()

                # Update FPS
                self.fps = self.clock.get_fps()
                self.clock.tick(60)

            except Exception as e:
                print(f"Error in main loop: {e}")
                continue

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    simulation = NanoHandSimulation()
    simulation.run()
