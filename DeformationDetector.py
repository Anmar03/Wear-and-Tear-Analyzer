import cv2
import numpy as np

class DeformationDetector:
    def __init__(self, image, window_name, screen_width, screen_height):
        self.image = image
        self.window_name = window_name
        self.screen_width = screen_width
        self.screen_height = screen_height

    def clear_screen(self):
        """Clears the screen by displaying a black image in the specified window."""
        black_image = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)  # Create a black image
        cv2.imshow(self.window_name, black_image)  # Display the black image to "clear" the window

    def run(self):
        """Main function to process the image and analyze the shoes."""
        # Load and resize the image
        self.image = cv2.imread('./TwoShoes.jpg')
        self.image = cv2.resize(self.image, (self.screen_width, self.screen_height))

        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)  

        # Detect edges and apply morphological closing
        edges = cv2.Canny(blurred, 50, 180)  
        kernel = np.ones((5, 5), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on aspect ratio and area
        filtered_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            if 1.5 < aspect_ratio < 4.5 and area > 3000:
                filtered_contours.append(contour)

        # Keep only the largest two contours
        filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:2]

        # Check if two contours are found
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
            analysis_text.append(f"Shape similarity: {shape_similarity:.4f}")

            if shape_similarity < 0.1:
                analysis_text.append("The shoes are very similar in shape.")
            elif 0.1 <= shape_similarity <= 0.5:
                analysis_text.append("The shoes have a moderate similarity in shape.")
            else:
                analysis_text.append("The shoes are not very similar in shape.")

            # Calculate areas
            area1 = cv2.contourArea(contour1)
            area2 = cv2.contourArea(contour2)
            analysis_text.append(f"Area of Shoe 1: {area1:.1f}")
            analysis_text.append(f"Area of Shoe 2: {area2:.1f}")
            
            # Calculate size difference
            if area1 > area2:
                percentage_diff = ((area1 - area2) / area2) * 100
                analysis_text.append(f"Shoe 1 is {percentage_diff:.2f}% bigger than Shoe 2.")
            elif area2 > area1:
                percentage_diff = ((area2 - area1) / area1) * 100
                analysis_text.append(f"Shoe 2 is {percentage_diff:.2f}% bigger than Shoe 1.")
            else:
                analysis_text.append("Both shoes are the same size.")

            # Draw contours on the original image
            cv2.drawContours(self.image, [contour1], -1, (0, 0, 255), 2)  # Red
            cv2.drawContours(self.image, [contour2], -1, (255, 0, 0), 2)  # Blue

            # Overlay the analysis text onto the image
            y0, dy = 50, 30  # Starting position and line spacing
            for i, line in enumerate(analysis_text):
                y = y0 + i * dy
                cv2.putText(self.image, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Show the image with contours and analysis
            cv2.imshow(self.window_name, self.image)
            cv2.waitKey(0)

            # Overlay two shoes to see differences
            overlay_canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)  # Create overlay canvas

            # Get bounding rectangles for contours
            x1, y1, w1, h1 = cv2.boundingRect(contour1)
            x2, y2, w2, h2 = cv2.boundingRect(contour2)

            # Shift contours for overlaying
            shift_x = 100  # Horizontal shift
            shift_y = 200  # Vertical shift
            contour1_shifted = contour1 - [x1 - shift_x, y1 - shift_y]
            contour2_shifted = contour2 - [x2 - shift_x, y2 - shift_y]

            # Draw shifted contours on the overlay canvas
            cv2.drawContours(overlay_canvas, [contour1_shifted], -1, (0, 0, 255), 1)  # Red for contour1
            cv2.drawContours(overlay_canvas, [contour2_shifted], -1, (255, 0, 0), 1)  # Blue for contour2

            # Display the overlaid contours
            cv2.imshow(self.window_name, overlay_canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            # If two shoes are not detected, display a message on the image
            message = "Could not detect two shoes in the image."
            cv2.putText(self.image, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow(self.window_name, self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
