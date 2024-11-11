import cv2
import numpy as np

class DeformationDetector:
    def __init__(self, image, window_name, screen_width, screen_height):
        self.image = image
        self.original_image = image.copy() if image is not None else None
        self.window_name = window_name
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.analysis_steps = []  # List of (image, text) tuples
        self.current_step = 0
        self.exit_analysis = False  # Flag to exit analysis and return to main menu

    def set_image(self, image):
        """Sets the image to be analyzed."""
        self.image = image
        self.original_image = image.copy() if image is not None else None

    def clear_screen(self):
        """Clears the screen by displaying a black image in the specified window."""
        black_image = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)  # Create a black image
        cv2.imshow(self.window_name, black_image)  # Display the black image to "clear" the window

    def run(self):
        """Main function to process the image and analyze the shoes."""
        if self.image is None:
            print("No image to analyze.")
            return

        self.exit_analysis = False  # Reset the flag
        self.analysis_steps = []  # Reset analysis steps
        self.current_step = 0  # Start from the first step

        # Resize the image to fit the screen
        h, w = self.image.shape[:2]
        scale = min(1.0, self.screen_width / w, self.screen_height / h)  # Prevent scaling up beyond original size
        new_w = int(w * scale)
        new_h = int(h * scale)
        self.image = cv2.resize(self.image, (new_w, new_h))
        self.original_image = self.image.copy()

        # Padding to center the image
        pad_w = max((self.screen_width - new_w) // 2, 0)
        pad_h = max((self.screen_height - new_h) // 2, 0)
        self.image = cv2.copyMakeBorder(
            self.image, pad_h, self.screen_height - new_h - pad_h,
            pad_w, self.screen_width - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        self.original_image = self.image.copy()

        # Step 1: Original Image
        analysis_text = ["Original Image"]
        self.analysis_steps.append((self.original_image.copy(), analysis_text))

        # Step 2: Grayscale and Blurred Image
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        blurred_image = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        analysis_text = ["Grayscale and Blurred Image"]
        self.analysis_steps.append((blurred_image, analysis_text))

        # Edge Detection and Contour Finding 
        edges = cv2.Canny(blurred, 50, 180)
        kernel = np.ones((5, 5), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on aspect ratio and area
        filtered_contours = []
        for contour in contours:
            x, y, w_contour, h_contour = cv2.boundingRect(contour)
            aspect_ratio = w_contour / float(h_contour) if h_contour != 0 else 0
            area = cv2.contourArea(contour)
            if 1.5 < aspect_ratio < 4.5 and area > 3000:
                filtered_contours.append(contour)

        # Keep only the largest two contours
        filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:2]

        # Final Analysis
        if len(filtered_contours) == 2:
            contour1 = filtered_contours[0]
            contour2 = filtered_contours[1]

            # Retain more of the shoe's shape details
            epsilon1 = 0.0001 * cv2.arcLength(contour1, True)
            epsilon2 = 0.0001 * cv2.arcLength(contour2, True)
            contour1 = cv2.approxPolyDP(contour1, epsilon1, True)
            contour2 = cv2.approxPolyDP(contour2, epsilon2, True)

            # Calculate shape similarity
            shape_similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)

            # Prepare analysis text
            analysis_text = []
            #analysis_text.append(f"Shape similarity: {shape_similarity:.4f}")

            if shape_similarity < 0.1:
                analysis_text.append("The shoes are very similar in shape.")
            elif 0.1 <= shape_similarity <= 0.5:
                analysis_text.append("The shoes have a moderate similarity in shape.")
            else:
                analysis_text.append("The shoes are not very similar in shape.")

            # Calculate areas
            area1 = cv2.contourArea(contour1)
            area2 = cv2.contourArea(contour2)
            #analysis_text.append(f"Area of Red Shoe: {area1:.1f}")
            #analysis_text.append(f"Area of Blue Shoe: {area2:.1f}")

            # Calculate size difference and provide meaning
            if area1 > area2:
                percentage_diff = ((area1 - area2) / area2) * 100
                analysis_text.append(f"Red Shoe is {percentage_diff:.2f}% bigger than Blue Shoe.")
                analysis_text.append("Therefore, Red Shoe is newer than the Blue Shoe.")
            elif area2 > area1:
                percentage_diff = ((area2 - area1) / area1) * 100
                analysis_text.append(f"Blue Shoe is {percentage_diff:.2f}% bigger than Red Shoe.")
                analysis_text.append("Therefore, Blue Shoe is newer than the Red Shoe.")
            else:
                analysis_text.append("Both shoes are the same size.")

            # Draw contours on the original image
            result_image = self.original_image.copy()
            cv2.drawContours(result_image, [contour1], -1, (0, 0, 255), 2)  # Red for contour1
            cv2.drawContours(result_image, [contour2], -1, (255, 0, 0), 2)  # Blue for contour2

            # Add labels to the contours
            x1, y1, w1, h1 = cv2.boundingRect(contour1)
            cv2.putText(result_image, "Red Shoe", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            x2, y2, w2, h2 = cv2.boundingRect(contour2)
            cv2.putText(result_image, "Blue Shoe", (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Overlay the analysis text onto the image
            self.analysis_steps.append((result_image, analysis_text))

            # Overlay Comparison
            overlay_canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

            # Calculate moments to find the center of the contours
            M1 = cv2.moments(contour1)
            M2 = cv2.moments(contour2)

            if M1["m00"] != 0 and M2["m00"] != 0:
                # Center coordinates of contours
                cX1 = int(M1["m10"] / M1["m00"])
                cY1 = int(M1["m01"] / M1["m00"])
                cX2 = int(M2["m10"] / M2["m00"])
                cY2 = int(M2["m01"] / M2["m00"])

                # Center of the overlay canvas
                canvas_center_x = self.screen_width // 2
                canvas_center_y = self.screen_height // 2

                # Shift contours to center them on the overlay canvas
                shift_x1 = canvas_center_x - cX1
                shift_y1 = canvas_center_y - cY1
                contour1_shifted = contour1 + [shift_x1, shift_y1]

                shift_x2 = canvas_center_x - cX2
                shift_y2 = canvas_center_y - cY2
                contour2_shifted = contour2 + [shift_x2, shift_y2]

                # Draw shifted contours on the overlay canvas
                cv2.drawContours(overlay_canvas, [contour1_shifted], -1, (0, 0, 255), 2)  # Red for contour1
                cv2.drawContours(overlay_canvas, [contour2_shifted], -1, (255, 0, 0), 2)  # Blue for contour2

                analysis_text = ["Overlay Comparison"]
                self.analysis_steps.append((overlay_canvas, analysis_text))
            else:
                # If moments are zero, cannot compute centroids
                message = "Could not center contours due to zero moments."
                cv2.putText(overlay_canvas, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                analysis_text = ["Overlay Comparison", message]
                self.analysis_steps.append((overlay_canvas, analysis_text))
        else:
            # If two shoes are not detected, display a message
            message_image = self.original_image.copy()
            message = "Could not detect two shoes in the image."
            cv2.putText(message_image, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            analysis_text = [message]
            self.analysis_steps.append((message_image, analysis_text))

        # Start displaying the analysis steps
        self.display_current_step()

    def display_current_step(self):
        """Displays the current analysis step with navigation arrows."""
        if self.current_step < 0:
            self.current_step = 0
        if self.current_step >= len(self.analysis_steps):
            self.current_step = len(self.analysis_steps) - 1

        # Get the current image and text
        image, analysis_text = self.analysis_steps[self.current_step]
        display_image = image.copy()

        # Overlay the analysis text onto the image
        y0, dy = 50, 30  # Starting position and line spacing
        for i, line in enumerate(analysis_text):
            y = y0 + i * dy
            cv2.putText(display_image, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Draw navigation arrows
        arrow_size = 50
        arrow_thickness = 2
        arrow_color = (255, 255, 255)

        # Left arrow (back)
        left_arrow_pts = np.array([
            [50, self.screen_height // 2],
            [50 + arrow_size, self.screen_height // 2 - arrow_size // 2],
            [50 + arrow_size, self.screen_height // 2 + arrow_size // 2]
        ], np.int32)
        cv2.fillPoly(display_image, [left_arrow_pts], arrow_color)
        self.left_arrow_coords = (50, self.screen_height // 2 - arrow_size // 2,
                                  50 + arrow_size, self.screen_height // 2 + arrow_size // 2)

        # Right arrow (forward)
        right_arrow_pts = np.array([
            [self.screen_width - 50, self.screen_height // 2],
            [self.screen_width - 50 - arrow_size, self.screen_height // 2 - arrow_size // 2],
            [self.screen_width - 50 - arrow_size, self.screen_height // 2 + arrow_size // 2]
        ], np.int32)
        cv2.fillPoly(display_image, [right_arrow_pts], arrow_color)
        self.right_arrow_coords = (self.screen_width - 50 - arrow_size, self.screen_height // 2 - arrow_size // 2,
                                   self.screen_width - 50, self.screen_height // 2 + arrow_size // 2)

        # Draw "Back to Menu" button
        button_width = 300
        button_height = 50
        button_x = (self.screen_width - button_width) // 2
        button_y = self.screen_height - 100
        self.back_button_coords = (button_x, button_y, button_x + button_width, button_y + button_height)
        cv2.rectangle(display_image, (self.back_button_coords[0], self.back_button_coords[1]),
                      (self.back_button_coords[2], self.back_button_coords[3]), (0, 0, 0), -1)
        cv2.rectangle(display_image, (self.back_button_coords[0], self.back_button_coords[1]),
                      (self.back_button_coords[2], self.back_button_coords[3]), (255, 255, 255), 2)
        text = "Back to Menu"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        text_x = self.back_button_coords[0] + (button_width - text_width) // 2
        text_y = self.back_button_coords[1] + (button_height + text_height) // 2 - 5
        cv2.putText(display_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Show the image
        cv2.imshow(self.window_name, display_image)

        # Set mouse callback for navigation
        cv2.setMouseCallback(self.window_name, self.analysis_mouse_callback)

        # Wait for key press or mouse event
        while True:
            key = cv2.waitKey(1)
            if self.exit_analysis:
                break
            if key == 27:  # ESC key to exit analysis
                self.exit_analysis = True
                break
        # After analysis is done, clear the screen and return to main menu
        self.clear_screen()

    def analysis_mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left arrow click area
            if self.left_arrow_coords[0] <= x <= self.left_arrow_coords[2] and self.left_arrow_coords[1] <= y <= self.left_arrow_coords[3]:
                self.current_step -= 1
                if self.current_step < 0:
                    self.current_step = 0
                self.display_current_step()
            # Right arrow click area
            elif self.right_arrow_coords[0] <= x <= self.right_arrow_coords[2] and self.right_arrow_coords[1] <= y <= self.right_arrow_coords[3]:
                self.current_step += 1
                if self.current_step >= len(self.analysis_steps):
                    self.current_step = len(self.analysis_steps) - 1
                self.display_current_step()
            # "Back to Menu" button click area
            elif self.back_button_coords[0] <= x <= self.back_button_coords[2] and self.back_button_coords[1] <= y <= self.back_button_coords[3]:
                self.exit_analysis = True
