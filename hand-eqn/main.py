import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Global variables for sliders
charge_strength = 50  # Default charge strength
particle_mass = 50    # Default particle mass
simulation_speed = 50 # Default simulation speed
gravity_strength = 50 # Default gravity strength
tunneling_probability = 5  # Default tunneling probability (percentage)

# Particle trails storage
particle_trails = []

def update_charge(value):
    global charge_strength
    charge_strength = value

def update_mass(value):
    global particle_mass
    particle_mass = value

def update_speed(value):
    global simulation_speed
    simulation_speed = value

def update_gravity(value):
    global gravity_strength
    gravity_strength = value

def update_tunneling(value):
    global tunneling_probability
    tunneling_probability = value

def get_finger_distance(hand_landmarks):
    """
    Calculate the distance between the thumb tip and index finger tip.
    """
    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
    distance = np.linalg.norm(thumb_tip - index_tip)
    return distance

def draw_proton(frame, hand_landmarks, radius, frame_width, frame_height):
    """
    Draw an enhanced proton with physics-based effects.
    """
    global particle_trails

    # Get the positions of the thumb tip and index finger tip
    thumb_tip = (int(hand_landmarks.landmark[4].x * frame_width), int(hand_landmarks.landmark[4].y * frame_height))
    index_tip = (int(hand_landmarks.landmark[8].x * frame_width), int(hand_landmarks.landmark[8].y * frame_height))

    # Calculate the midpoint between thumb and index finger
    midpoint = ((thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2)

    # Draw a glowing halo around the proton (energy barrier)
    halo_radius = int(radius * 150)
    halo_thickness = int(10 + 5 * np.sin(cv2.getTickCount() / 1e6))  # Pulsating effect
    cv2.circle(frame, midpoint, halo_radius, (0, 165, 255), thickness=halo_thickness)  # Orange halo
    cv2.circle(frame, midpoint, halo_radius, (0, 0, 255), thickness=2)      # Inner red glow

    # Draw the proton core with 3D effect (gradient + shadow)
    core_radius = int(radius * 100)
    overlay = frame.copy()
    cv2.circle(overlay, midpoint, core_radius, (0, 0, 255), -1)  # Red core
    cv2.circle(overlay, midpoint, core_radius // 2, (255, 255, 0), -1)  # Yellow inner core

    # Add shadow for 3D effect
    shadow_offset = 10
    cv2.circle(frame, (midpoint[0] + shadow_offset, midpoint[1] + shadow_offset), core_radius, (50, 50, 50), -1)

    # Blend the core with transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Add Bohr model energy levels
    for i in range(1, 4):  # Draw 3 energy levels
        level_radius = core_radius + i * 20
        cv2.circle(frame, midpoint, level_radius, (255, 255, 255), thickness=1)

    # Add electric field lines
    for angle in np.linspace(0, 2 * np.pi, 16):  # 16 radial lines
        end_x = int(midpoint[0] + halo_radius * np.cos(angle))
        end_y = int(midpoint[1] + halo_radius * np.sin(angle))
        cv2.line(frame, midpoint, (end_x, end_y), (255, 255, 255), thickness=1)

    # Add orbiting particles
    num_particles = 5
    for i in range(num_particles):
        angle = (cv2.getTickCount() / (1 + simulation_speed)) % (2 * np.pi) + i * (2 * np.pi / num_particles)
        orbit_radius = core_radius + 50
        x = int(midpoint[0] + orbit_radius * np.cos(angle))
        y = int(midpoint[1] + orbit_radius * np.sin(angle))

        # Check for quantum tunneling
        if np.random.rand() < tunneling_probability / 100:  # Random chance for tunneling
            # Teleport particle outside the energy barrier
            tunnel_angle = np.random.uniform(0, 2 * np.pi)
            tunnel_radius = halo_radius + np.random.uniform(10, 50)
            x = int(midpoint[0] + tunnel_radius * np.cos(tunnel_angle))
            y = int(midpoint[1] + tunnel_radius * np.sin(tunnel_angle))
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Blue dot for tunneled particle
        else:
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)  # White dot for normal particle

        # Store particle trail
        particle_trails.append((x, y))
        if len(particle_trails) > 50:  # Limit trail length
            particle_trails.pop(0)

    # Draw particle trails
    for idx, (trail_x, trail_y) in enumerate(particle_trails):
        alpha = 1 - idx / len(particle_trails)  # Fade effect
        color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
        cv2.circle(frame, (trail_x, trail_y), 2, color, -1)

    # Add text annotations
    cv2.putText(frame, f"Charge: +{charge_strength}", (midpoint[0] - 50, midpoint[1] - halo_radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Mass: {particle_mass}", (midpoint[0] - 50, midpoint[1] - halo_radius + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Tunneling: {tunneling_probability}%", (midpoint[0] - 50, midpoint[1] - halo_radius + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return midpoint, core_radius

def simulate_physics(protons, frame):
    """
    Simulate Coulomb's law, gravity, and particle interactions.
    """
    for i, (p1_pos, p1_radius) in enumerate(protons):
        for j, (p2_pos, p2_radius) in enumerate(protons):
            if i != j:
                # Calculate distance between two protons
                dx = p2_pos[0] - p1_pos[0]
                dy = p2_pos[1] - p1_pos[1]
                distance = np.sqrt(dx**2 + dy**2)

                # Repulsion force (Coulomb-like repulsion)
                if distance > 0 and distance < (p1_radius + p2_radius) * 2:
                    repulsion_force = charge_strength / distance  # Inverse relationship
                    dx_norm = dx / distance
                    dy_norm = dy / distance

                    # Move protons apart
                    p1_pos[0] -= int(repulsion_force * dx_norm)
                    p1_pos[1] -= int(repulsion_force * dy_norm)
                    p2_pos[0] += int(repulsion_force * dx_norm)
                    p2_pos[1] += int(repulsion_force * dy_norm)

        # Simulate gravity (attraction toward the center of the screen)
        screen_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        dx = screen_center[0] - p1_pos[0]
        dy = screen_center[1] - p1_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        if distance > 0:
            gravity_force = gravity_strength / distance  # Inverse relationship
            dx_norm = dx / distance
            dy_norm = dy / distance
            p1_pos[0] += int(gravity_force * dx_norm)
            p1_pos[1] += int(gravity_force * dy_norm)

# Create a window with sliders
cv2.namedWindow("Settings")
cv2.createTrackbar("Charge Strength", "Settings", charge_strength, 100, update_charge)
cv2.createTrackbar("Particle Mass", "Settings", particle_mass, 100, update_mass)
cv2.createTrackbar("Simulation Speed", "Settings", simulation_speed, 100, update_speed)
cv2.createTrackbar("Gravity Strength", "Settings", gravity_strength, 100, update_gravity)
cv2.createTrackbar("Tunneling Probability", "Settings", tunneling_probability, 100, update_tunneling)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a more natural view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)

    protons = []  # List to store proton positions and radii

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the distance between thumb and index finger
            distance = get_finger_distance(hand_landmarks)

            # Map the distance to a proton radius (adjust scaling as needed)
            proton_radius = max(0.05, min(0.5, distance * 2))

            # Draw the proton and get its position
            midpoint, core_radius = draw_proton(frame, hand_landmarks, proton_radius, frame.shape[1], frame.shape[0])
            protons.append([list(midpoint), core_radius])

    # Simulate physics between protons
    simulate_physics(protons, frame)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
