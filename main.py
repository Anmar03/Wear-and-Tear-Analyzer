import cv2
import numpy as np


class StainDetector:
    shoes_cropped = []
    overlay = None
    similarity = 0
    similarity_text = ''


    def __init__(self, image_path, window_name='Shoes', screen_width=1920, screen_height=1080):
        shoes = self.extract_shoes(image_path)

        self.compare_shoes(shoes)
        self.overlay = self.overlay_stains(self.shoes_cropped)
        self.display(screen_width, screen_height)
        

    # Function to extract shoes from an image
    def extract_shoes(self, image_path):
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        image = cv2.resize(image, (int(width * 0.3), int(height * 0.3)))  
        
        # Convert to grayscale for thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Canny edge detection
        edges = cv2.Canny(filtered, 53, 139)

        # Morphological closing to close gaps between edges
        kernel = np.ones((5, 5), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours (shoe shapes)
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            if 1.5 < aspect_ratio < 4.5 and area > 3000:
                filtered_contours.append(contour)

        # Sort contours by area and keep the two largest ones (which should correspond to the shoes)
        filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:2]

        if len(filtered_contours) != 2:
            print("Shoes not found")
            exit()

        # Masks for the two shoes
        shoe_masks = [np.zeros_like(gray) for _ in filtered_contours]

        # Extract the two shoes as ROIs
        shoes = []
        for i, contour in enumerate(filtered_contours):
            cv2.drawContours(shoe_masks[i], [contour], -1, 255, thickness=cv2.FILLED)
            shoe = cv2.bitwise_and(image, image, mask=shoe_masks[i])

            x, y, w, h = cv2.boundingRect(contour)
            shoe_cropped = shoe[y:y+h, x:x+w]
            self.shoes_cropped.append(shoe_cropped)

            shoes.append(shoe)
        
        return shoes
            

    # Function to calculate color histogram for each shoe
    def calc_histogram(self, roi):
        # Convert to HSV for better color comparison
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Calculate the histogram for each channel (H, S, V)
        hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [100, 120, 120], [0, 180, 0, 256, 0, 256])
        # Normalize the histogram for better comparison
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    
    def compare_shoes(self, shoes):
        # Calculate histograms for both shoes
        hist1 = self.calc_histogram(shoes[0])
        hist2 = self.calc_histogram(shoes[1])
        max_threshold = 100.0

        # Compare histograms using chi-square method 
        self.similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        self.similarity = max(0, (1 - self.similarity / max_threshold) * 100)

        high_similarity = 80
        moderate_similarity = 60

        if self.similarity >= high_similarity:
            self.similarity_text = "Highly similar"
            print(f"The shoes are highly similar in color with a similarity of {self.similarity:.2f}%")
        elif self.similarity >= moderate_similarity:
            self.similarity_text = "Moderately similar"
            print(f"The shoes have moderate color similarity with a similarity of {self.similarity:.2f}%")
        else:
            self.similarity_text = "Not similar"
            print(f"The shoes are not similar in color with a similarity of {self.similarity:.2f}%")

        
    def overlay_stains(self, shoes):
        hsv_shoe1 = cv2.cvtColor(shoes[0], cv2.COLOR_BGR2HSV)
        hsv_shoe2 = cv2.cvtColor(shoes[1], cv2.COLOR_BGR2HSV)

        hsv_shoe1 = cv2.resize(hsv_shoe1, (hsv_shoe2.shape[1], hsv_shoe2.shape[0]))

        diff = cv2.absdiff(hsv_shoe1, hsv_shoe2)

        # Converting differences to grayscale, highlighting significant areas
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Thresholding differences to focus on major color changes
        _, diff_thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        
        # Creating a heatmap to visualise the difference
        heatmap = cv2.applyColorMap(diff_thresh, cv2.COLORMAP_JET)

        # Resizing heatmap to match original size of shoes
        heatmap = cv2.resize(heatmap, (shoes[0].shape[1], shoes[0].shape[0]))
        
        # Blending heatmap with first shoe image
        overlay = cv2.addWeighted(shoes[0], 0.7, heatmap, 0.3, 0)
        
        return overlay

    
    def display(self, screen_width, screen_height):
        def combine_images_vertically(image1, image2):
            # Ensuring both images have the same width for vertical stacking
            width1 = image1.shape[1]
            width2 = image2.shape[1]
            target_width = min(width1, width2)

            # Scale images to have the same width
            scale1 = target_width / width1
            scale2 = target_width / width2
            new_height1 = int(image1.shape[0] * scale1)
            new_height2 = int(image2.shape[0] * scale2)
            resized_image1 = cv2.resize(image1, (target_width, new_height1))
            resized_image2 = cv2.resize(image2, (target_width, new_height2))

            # Combine the two canvases vertically
            combined_image = np.vstack((resized_image1, resized_image2))
            return combined_image


        def show_fullscreen(window_name, image, text, center=True):
            # Calculate the maximum height for the image (taking the screen height into account)
            available_height = screen_height  # Reserve space for the text field
            height, width = image.shape[:2]

            # Calculate the scale factor based on the height (to maximize height usage)
            scale = available_height / height

            # Recalculate the width based on the scaling factor to maintain the aspect ratio
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            padding = 20
            font_scale = 2
            thickness = 2

            # Resize the image to fit the calculated dimensions
            resized_image = cv2.resize(image, (new_width, new_height))

            # Create a canvas that can accommodate the image and the text
            canvas_height = new_height + 100 + padding  # Space for the image and text

            if center:
                canvas_width = screen_width  # Use full screen width if centering
            else:
                canvas_width = new_width

            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            # Calculate the left padding to center the image horizontally
            left_padding = (canvas_width - new_width) // 2

            if center:
                # Place the resized image on the canvas, centered horizontally
                canvas[:new_height, left_padding:left_padding + new_width] = resized_image
            else:
                canvas[:new_height, :] = resized_image

            # Add text below the image
            text_position = (left_padding - len(text), new_height + padding + 40)  # Adjust text position
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, text, text_position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Create a window and set it to fullscreen
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # Display the canvas with the image and text
            cv2.imshow(window_name, canvas)


        combined_image = combine_images_vertically(self.shoes_cropped[0], self.shoes_cropped[1])

        # Display the shoes and overlay
        show_fullscreen('Shoes', combined_image, self.similarity_text + f' Similarity: {self.similarity:.2f}' + '%', center=True)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        show_fullscreen('Overlay', self.overlay, 'Shoe stains overlayed on Shoe 1', center=False)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        


# Test the class
stain_detector = StainDetector('shoes.jpg', 'Shoes', 1920, 1080)