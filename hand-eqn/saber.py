import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import math
import random
import os

class EnhancedValentineCard:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Create output directory
        self.output_dir = "valentine_cards"
        os.makedirs(self.output_dir, exist_ok=True)

        # Card settings
        self.card_width = 800
        self.card_height = 600
        self.current_template = 0
        self.current_filter = 0
        self.current_border = 0
        self.show_frame_preview = True

        # Initialize states
        self.card_saved = False
        self.countdown_active = False
        self.countdown = 3
        self.last_countdown_time = 0
        self.preview_card = None

        # Templates with borders
        self.borders = [
            "None",
            "Hearts",
            "Stars",
            "Double Line",
            "Vintage"
        ]

        # Templates
        self.templates = [
            {
                "name": "Classic Red",
                "bg_color": np.array([255, 255, 255]),
                "text_color": np.array([0, 0, 255]),
                "accent_color": np.array([0, 0, 200])
            },
            {
                "name": "Romantic Pink",
                "bg_color": np.array([235, 206, 255]),
                "text_color": np.array([147, 20, 255]),
                "accent_color": np.array([180, 105, 255])
            },
            {
                "name": "Royal Purple",
                "bg_color": np.array([255, 240, 245]),
                "text_color": np.array([128, 0, 128]),
                "accent_color": np.array([147, 112, 219])
            },
            {
                "name": "Golden Love",
                "bg_color": np.array([240, 248, 255]),
                "text_color": np.array([184, 134, 11]),
                "accent_color": np.array([218, 165, 32])
            }
        ]

        # Filters
        self.filters = [
            ("Normal", lambda img: img),
            ("Warm", self.apply_warm_filter),
            ("Cool", self.apply_cool_filter),
            ("Vintage", self.apply_vintage_filter),
            ("B&W Hearts", self.apply_bw_with_hearts),
            ("Dreamy", self.apply_dreamy_filter)
        ]

        # Messages
        self.messages = [
            "Happy Valentine's Day!",
            "Be My Valentine",
            "Love is in the Air",
            "Forever Yours",
            "XOXO"
        ]
        self.current_message = 0

    def apply_warm_filter(self, image):
        """Apply warm temperature filter"""
        img = image.copy()
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=15)
        img[:, :, 0] = cv2.add(img[:, :, 0], -30)  # Reduce blue
        img[:, :, 2] = cv2.add(img[:, :, 2], 30)   # Increase red
        return img

    def apply_cool_filter(self, image):
        """Apply cool temperature filter"""
        img = image.copy()
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
        img[:, :, 0] = cv2.add(img[:, :, 0], 30)   # Increase blue
        img[:, :, 2] = cv2.add(img[:, :, 2], -20)  # Reduce red
        return img

    def apply_vintage_filter(self, image):
        """Apply vintage filter with vignette effect"""
        img = image.copy()

        # Apply sepia tone
        img = cv2.convertScaleAbs(img, alpha=1.3, beta=20)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Create vignette effect
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/4)
        kernel_y = cv2.getGaussianKernel(rows, rows/4)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)

        for i in range(3):
            img[:, :, i] = img[:, :, i] * mask

        return img

    def apply_dreamy_filter(self, image):
        """Apply dreamy soft focus effect"""
        img = image.copy()

        # Create soft focus effect
        blur = cv2.GaussianBlur(img, (0, 0), 10)
        img = cv2.addWeighted(img, 0.75, blur, 0.25, 0)

        # Enhance colors slightly
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)

        return img

    def apply_bw_with_hearts(self, image):
        """Convert to B&W but keep red/pink hearts in color"""
        img = image.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Create mask for red/pink colors
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([160, 100, 100])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2

        # Combine grayscale and color where red is detected
        result = gray.copy()
        result[mask > 0] = img[mask > 0]

        return result

    def draw_border(self, card, style):
        """Draw decorative border based on style"""
        if style == "None":
            return card

        elif style == "Hearts":
            spacing = 50
            size = 15
            for i in range(0, self.card_width, spacing):
                self.draw_heart(card, i, 20, size, (0, 0, 255))
                self.draw_heart(card, i, self.card_height-20, size, (0, 0, 255))
            for i in range(0, self.card_height, spacing):
                self.draw_heart(card, 20, i, size, (0, 0, 255))
                self.draw_heart(card, self.card_width-20, i, size, (0, 0, 255))

        elif style == "Stars":
            padding = 20
            cv2.rectangle(card,
                         (padding, padding),
                         (self.card_width-padding, self.card_height-padding),
                         (0, 0, 255), 2)
            # Draw stars at corners
            star_points = [
                (padding, padding),
                (self.card_width-padding, padding),
                (padding, self.card_height-padding),
                (self.card_width-padding, self.card_height-padding)
            ]
            for x, y in star_points:
                self.draw_star(card, x, y, 15, (0, 0, 255))

        elif style == "Double Line":
            padding1, padding2 = 15, 25
            cv2.rectangle(card,
                         (padding1, padding1),
                         (self.card_width-padding1, self.card_height-padding1),
                         (0, 0, 255), 1)
            cv2.rectangle(card,
                         (padding2, padding2),
                         (self.card_width-padding2, self.card_height-padding2),
                         (0, 0, 255), 1)

        elif style == "Vintage":
            padding = 20
            # Draw main rectangle
            cv2.rectangle(card,
                         (padding, padding),
                         (self.card_width-padding, self.card_height-padding),
                         (0, 0, 255), 2)
            # Draw corner decorations
            corner_size = 30
            corners = [
                ((padding, padding), (padding+corner_size, padding), (padding, padding+corner_size)),
                ((self.card_width-padding-corner_size, padding), (self.card_width-padding, padding),
                 (self.card_width-padding, padding+corner_size)),
                ((padding, self.card_height-padding-corner_size), (padding, self.card_height-padding),
                 (padding+corner_size, self.card_height-padding)),
                ((self.card_width-padding, self.card_height-padding-corner_size),
                 (self.card_width-padding, self.card_height-padding),
                 (self.card_width-padding-corner_size, self.card_height-padding))
            ]
            for corner in corners:
                cv2.line(card, corner[0], corner[1], (0, 0, 255), 2)
                cv2.line(card, corner[1], corner[2], (0, 0, 255), 2)

        return card

    def draw_heart(self, img, x, y, size, color):
        """Draw a heart shape"""
        points = []
        for t in np.linspace(0, 2*math.pi, 30):
            r = size * (1 - math.sin(t))
            points.append((
                int(x + r * math.cos(t)),
                int(y + r * math.sin(t))
            ))
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(img, [pts], color)

    def draw_star(self, img, x, y, size, color):
        """Draw a star shape"""
        points = []
        for i in range(5):
            angle = i * 4 * math.pi / 5 - math.pi / 2
            points.append((
                int(x + size * math.cos(angle)),
                int(y + size * math.sin(angle))
            ))
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(img, [pts], color)

    def create_valentine_card(self, frame):
        """Create a Valentine's card with the captured frame"""
        try:
            template = self.templates[self.current_template]

            # Create base card
            card = np.full((self.card_height, self.card_width, 3),
                          template["bg_color"], dtype=np.uint8)

            # Apply current filter to frame
            filtered_frame = self.filters[self.current_filter][1](frame)

            # Resize and position frame
            frame_height = int(self.card_height * 0.6)
            frame_width = int(frame_height * frame.shape[1] / frame.shape[0])
            frame_resized = cv2.resize(filtered_frame, (frame_width, frame_height))

            if len(frame_resized.shape) == 2:
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2BGR)

            # Add frame to card
            x_offset = (self.card_width - frame_width) // 2
            y_offset = int(self.card_height * 0.25)  # Position at 25% from top

            card[y_offset:y_offset+frame_height,
                 x_offset:x_offset+frame_width] = frame_resized

            # Add border
            card = self.draw_border(card, self.borders[self.current_border])

            # Add message
            message = self.messages[self.current_message]
            text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                                      1.5, 2)[0]
            text_x = (self.card_width - text_size[0]) // 2
            cv2.putText(card, message,
                       (text_x, int(self.card_height * 0.15)),
                       cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                       1.5, tuple(map(int, template["text_color"])), 2)

            # Add date
            date_str = datetime.now().strftime("%B %d, %Y")
            cv2.putText(card, date_str,
                       (self.card_width//3, self.card_height-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       tuple(map(int, template["text_color"])), 1)

            return card

        except Exception as e:
            print(f"Error creating card: {str(e)}")
            return np.full((self.card_height, self.card_width, 3),
                          (255, 255, 255), dtype=np.uint8)

    def detect_heart_gesture(self, hand_landmarks1, hand_landmarks2):
        """Detect if hands are making a heart shape"""
        try:
            if hand_landmarks1 and hand_landmarks2:
                left_thumb = (hand_landmarks1.landmark[4].x, hand_landmarks1.landmark[4].y)
                left_index = (hand_landmarks1.landmark[8].x, hand_landmarks1.landmark[8].y)
                right_thumb = (hand_landmarks2.landmark[4].x, hand_landmarks2.landmark[4].y)
                right_index = (hand_landmarks2.landmark[8].x, hand_landmarks2.landmark[8].y)

                thumb_distance = math.sqrt((left_thumb[0] - right_thumb[0])**2 +
                                         (left_thumb[1] - right_thumb[1])**2)
                index_distance = math.sqrt((left_index[0] - right_index[0])**2 +
                                         (left_index[1] - right_index[1])**2)

                return thumb_distance < 0.15 and index_distance < 0.15
        except Exception as e:
            print(f"Error detecting gesture: {str(e)}")


    def update_preview(self, frame):
        """Update the preview card"""
        if self.show_frame_preview:
            self.preview_card = self.create_valentine_card(frame)

    def draw_ui_overlay(self, frame):
        """Draw UI elements on the frame"""
        # Create a semi-transparent overlay for text
        overlay = frame.copy()

        # Add current settings display
        settings_bg = np.zeros((150, 300, 3), dtype=np.uint8)
        cv2.putText(settings_bg, f"Template: {self.templates[self.current_template]['name']}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(settings_bg, f"Filter: {self.filters[self.current_filter][0]}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(settings_bg, f"Border: {self.borders[self.current_border]}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(settings_bg, f"Message: {self.messages[self.current_message]}",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add settings to main frame
        frame[20:170, 20:320] = cv2.addWeighted(frame[20:170, 20:320], 0.2, settings_bg, 0.8, 0)

        # Add instructions
        instructions = [
            "Controls:",
            "T - Change Template",
            "F - Change Filter",
            "B - Change Border",
            "M - Change Message",
            "P - Toggle Preview",
            "Q - Quit"
        ]

        y_pos = frame.shape[0] - 180
        for instruction in instructions:
            cv2.putText(frame, instruction,
                       (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25

        # Add gesture instruction
        cv2.putText(frame, "Make a heart shape with both hands!",
                   (frame.shape[1]//2 - 200, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        return frame

    def run(self):
        """Main application loop"""
        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    print("Failed to grab frame")
                    break

                # Flip frame horizontally for selfie-view
                frame = cv2.flip(frame, 1)

                # Process hand tracking
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                # Update preview if enabled
                if self.show_frame_preview and not self.countdown_active:
                    self.update_preview(frame)

                # Draw hand landmarks and process gesture
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2),
                            self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                        )

                    # Check for heart gesture
                    if len(results.multi_hand_landmarks) == 2:
                        if self.detect_heart_gesture(
                            results.multi_hand_landmarks[0],
                            results.multi_hand_landmarks[1]
                        ):
                            if not self.countdown_active:
                                self.countdown_active = True
                                self.countdown = 3
                                self.last_countdown_time = datetime.now().timestamp()

                # Handle countdown and capture
                if self.countdown_active:
                    current_time = datetime.now().timestamp()
                    if current_time - self.last_countdown_time >= 1:
                        self.countdown -= 1
                        self.last_countdown_time = current_time

                        if self.countdown == 0:
                            # Create and save card
                            card = self.create_valentine_card(frame)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join(self.output_dir, f'valentine_card_{timestamp}.jpg')
                            cv2.imwrite(filename, card)
                            print(f"Card saved as {filename}")

                            # Reset states
                            self.card_saved = True
                            self.countdown_active = False

                            # Show saved card briefly
                            cv2.imshow('Saved Card', card)
                            cv2.waitKey(2000)  # Show for 2 seconds
                            cv2.destroyWindow('Saved Card')


                    cv2.putText(frame, str(self.countdown),
                               (frame.shape[1]//2-50, frame.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 255), 4)


                frame = self.draw_ui_overlay(frame)


                cv2.imshow('Valentine Card Creator', frame)

                # Show preview if enabled
                if self.show_frame_preview and self.preview_card is not None:
                    cv2.imshow('Card Preview', self.preview_card)


                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.current_template = (self.current_template + 1) % len(self.templates)
                elif key == ord('f'):
                    self.current_filter = (self.current_filter + 1) % len(self.filters)
                elif key == ord('b'):
                    self.current_border = (self.current_border + 1) % len(self.borders)
                elif key == ord('m'):
                    self.current_message = (self.current_message + 1) % len(self.messages)
                elif key == ord('p'):
                    self.show_frame_preview = not self.show_frame_preview
                    if not self.show_frame_preview:
                        cv2.destroyWindow('Card Preview')

        except Exception as e:
            print(f"Error in main loop: {str(e)}")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Valentine Card Creator...")
    print("Controls:")
    print("T - Change Template")
    print("F - Change Filter")
    print("B - Change Border")
    print("M - Change Message")
    print("P - Toggle Preview")
    print("Q - Quit")

    card_creator = EnhancedValentineCard()
    card_creator.run()
