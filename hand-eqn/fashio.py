import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
import json

class SmartWardrobeAssistant:
    def __init__(self, clothing_dir='clothing'):
        # Initialize MediaPipe
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Enhanced pose detection with higher accuracy
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,  # Increased confidence threshold
            model_complexity=2,  # Using more complex model for better accuracy
            smooth_landmarks=True  # Enable landmark smoothing
        )

        # Enhanced hand detection
        self.hands = mp_hands.Hands(
            max_num_hands=2,  # Support for both hands
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6,
            model_complexity=1
        )

        # Load clothing items and metadata
        self.clothing_dir = clothing_dir
        self.clothing_data = self._load_clothing_data()
        self.current_clothing_index = 0

        # State tracking
        self.last_gesture_time = datetime.now()
        self.gesture_cooldown = 0.5  # seconds
        self.pose_history = []
        self.history_size = 5

        # Configure camera with high resolution
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def _load_clothing_data(self):
        """Load clothing items with metadata."""
        clothing_data = []
        metadata_file = os.path.join(self.clothing_dir, 'metadata.json')

        # Load or create metadata
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        if os.path.exists(self.clothing_dir):
            for filename in os.listdir(self.clothing_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(self.clothing_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                    if img is not None:
                        # Process image
                        if len(img.shape) == 2:  # Grayscale
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                        elif img.shape[2] == 3:  # RGB
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

                        # Get or create metadata for this item
                        item_metadata = metadata.get(filename, {
                            'name': os.path.splitext(filename)[0],
                            'category': 'unknown',
                            'position_offset': {'x': 0, 'y': 0},
                            'scale_factor': 1.0
                        })

                        clothing_data.append({
                            'image': img,
                            'metadata': item_metadata
                        })

        return clothing_data or [{'image': np.zeros((300, 200, 4), dtype=np.uint8), 'metadata': {'name': 'default'}}]

    def calculate_body_metrics(self, pose_landmarks, frame_shape):
        """Calculate detailed body measurements for better fitting."""
        metrics = {}
        if pose_landmarks:
            h, w = frame_shape[:2]

            # Shoulder width
            shoulder_left = pose_landmarks.landmark[11]
            shoulder_right = pose_landmarks.landmark[12]
            metrics['shoulder_width'] = abs(shoulder_right.x - shoulder_left.x) * w

            # Torso length
            neck = pose_landmarks.landmark[11]  # Using left shoulder as neck base
            hip = pose_landmarks.landmark[23]  # Left hip
            metrics['torso_length'] = abs(hip.y - neck.y) * h

            # Body angle
            spine_x = (shoulder_left.x + shoulder_right.x) / 2
            spine_y = (shoulder_left.y + hip.y) / 2
            metrics['body_angle'] = np.arctan2(spine_y, spine_x)

            # Body scale
            metrics['body_scale'] = metrics['shoulder_width'] / w

        return metrics

    def overlay_clothing(self, frame, pose_landmarks, clothing_item):
        """Enhanced clothing overlay with advanced positioning and blending."""
        if not pose_landmarks or not clothing_item:
            return frame

        image = clothing_item['image']
        metadata = clothing_item['metadata']

        # Get body measurements
        metrics = self.calculate_body_metrics(pose_landmarks, frame.shape)
        if not metrics:
            return frame

        # Calculate advanced positioning
        frame_h, frame_w = frame.shape[:2]

        # Apply dynamic scaling based on body metrics
        base_scale = metrics['body_scale'] * metadata['scale_factor']
        scale_factor = base_scale * 4.0  # Adjustable scaling factor

        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)

        # Resize clothing with maintain aspect ratio
        clothing_resized = cv2.resize(image, (new_width, new_height),
                                    interpolation=cv2.INTER_LANCZOS4)

        # Calculate position with metadata offsets
        center_x = int(((pose_landmarks.landmark[11].x + pose_landmarks.landmark[12].x) / 2 * frame_w) +
                      metadata['position_offset']['x'])
        center_y = int((pose_landmarks.landmark[11].y * frame_h) +
                      metadata['position_offset']['y'])

        # Apply positioning
        x1 = center_x - new_width // 2
        y1 = center_y - new_height // 2
        x2, y2 = x1 + new_width, y1 + new_height

        # Ensure within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)

        # Get ROI and adjust clothing section
        roi = frame[y1:y2, x1:x2]
        clothing_roi = clothing_resized[:roi.shape[0], :roi.shape[1]]

        # Enhanced alpha blending with edge smoothing
        mask = clothing_roi[:, :, 3] / 255.0
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = np.stack([mask, mask, mask], axis=2)

        # Apply color correction to match lighting
        clothing_color = clothing_roi[:, :, :3].astype(float)
        roi_color = roi.astype(float)

        # Simple color matching
        roi_mean = np.mean(roi_color, axis=(0, 1))
        clothing_mean = np.mean(clothing_color, axis=(0, 1))
        color_correction = roi_mean / (clothing_mean + 1e-6)
        clothing_color *= color_correction.reshape(1, 1, 3)
        clothing_color = np.clip(clothing_color, 0, 255)

        # Blend with smooth edges
        blended = (roi * (1 - mask) + clothing_color * mask).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended

        return frame

    def detect_gestures(self, hand_landmarks):
        """Enhanced gesture detection with multiple gesture support."""
        if not hand_landmarks:
            return None

        current_time = datetime.now()
        if (current_time - self.last_gesture_time).total_seconds() < self.gesture_cooldown:
            return None

        wrist = hand_landmarks.landmark[0]
        index_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]

        # Horizontal swipe detection
        horizontal_diff = index_tip.x - wrist.x
        vertical_diff = abs(index_tip.y - wrist.y)

        # Calculate velocity for more accurate gesture detection
        gesture = None
        if vertical_diff < 0.1:  # Ensure horizontal movement
            if horizontal_diff > 0.2:
                gesture = "right"
            elif horizontal_diff < -0.2:
                gesture = "left"

        # Pinch gesture for zoom
        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2
        )
        if thumb_index_dist < 0.05:
            gesture = "pinch"

        if gesture:
            self.last_gesture_time = current_time

        return gesture

    def run(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame. Exiting...")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process pose and hands
                pose_results = self.pose.process(rgb_frame)
                hand_results = self.hands.process(rgb_frame)

                # Handle gestures
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        gesture = self.detect_gestures(hand_landmarks)
                        if gesture == "left":
                            self.current_clothing_index = (self.current_clothing_index - 1) % len(self.clothing_data)
                        elif gesture == "right":
                            self.current_clothing_index = (self.current_clothing_index + 1) % len(self.clothing_data)

                # Overlay clothing with enhanced features
                if pose_results.pose_landmarks:
                    current_item = self.clothing_data[self.current_clothing_index]
                    frame = self.overlay_clothing(
                        frame,
                        pose_results.pose_landmarks,
                        current_item
                    )

                    # Draw UI elements
                    item_name = current_item['metadata']['name']
                    cv2.putText(
                        frame,
                        f"{item_name} ({self.current_clothing_index + 1}/{len(self.clothing_data)})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                cv2.imshow("Smart Wardrobe Assistant", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    break

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    wardrobe_app = SmartWardrobeAssistant()
    wardrobe_app.run()

if __name__ == "__main__":
    main()
