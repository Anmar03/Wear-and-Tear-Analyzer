import cv2
import numpy as np


class StainDetector:
    shoes_cropped = []
    similarity = 0


    def __init__(self, image_path):
        shoes = self.extract_shoes(image_path)

        self.compare_shoes(shoes)
        

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
            print(f"The shoes are highly similar in color with a similarity of {self.similarity:.2f}%.")
        elif self.similarity >= moderate_similarity:
            print(f"The shoes have moderate color similarity with a similarity of {self.similarity:.2f}%.")
        else:
            print(f"The shoes are not similar in color with a similarity of {self.similarity:.2f}%.")


# Test the class
image_path = 'shoes.jpg'
stain_detector = StainDetector(image_path)
shoes = stain_detector.shoes_cropped

scale_factor = 0.3
shoes[0] = cv2.resize(shoes[0], (int(shoes[0].shape[1] * scale_factor), int(shoes[0].shape[0] * scale_factor)))
shoes[1] = cv2.resize(shoes[1], (int(shoes[1].shape[1] * scale_factor), int(shoes[1].shape[0] * scale_factor)))

cv2.imshow('Shoe 1', shoes[0])
cv2.imshow('Shoe 2', shoes[1])
cv2.waitKey(0)
cv2.destroyAllWindows()