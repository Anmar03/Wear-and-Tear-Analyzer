import cv2
import numpy as np

def extract_shoes(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale for thresholding
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 180)

    # Use morphological closing to close gaps between edges
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours (shoe shapes)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and keep the two largest ones (which should correspond to the shoes)
    valid_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    if len(valid_contours) < 2:
            print("Shoes not found")
            exit()

    # Create masks for the two shoes
    shoe_masks = [np.zeros_like(gray_image) for _ in valid_contours]

    # Extract the two shoes as ROIs
    shoes = []
    for i, contour in enumerate(valid_contours):
        cv2.drawContours(shoe_masks[i], [contour], -1, 255, thickness=cv2.FILLED)
        shoe = cv2.bitwise_and(image, image, mask=shoe_masks[i])
        shoes.append(shoe)
    
    return shoes
        

# Function to calculate color histogram for each shoe
def calc_histogram(roi):
    # Convert to HSV for better color comparison
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Calculate the histogram for each channel (H, S, V)
    hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
    # Normalize the histogram for better comparison
    hist = cv2.normalize(hist, hist).flatten()
    return hist

shoes = extract_shoes('shoes.jpg')

# Calculate histograms for both shoes
hist1 = calc_histogram(shoes[0])
hist2 = calc_histogram(shoes[1])

# Compare histograms using correlation method (or you can try other methods like chi-square, intersection, etc.)
similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Set a threshold to account for lighting differences
threshold = 0.8  # You can adjust this based on acceptable lighting variations
if similarity > threshold:
    print(f"The shoes are {similarity * 100:.2f}% similar in color.")
else:
    print(f"The shoes have significant color differences: {similarity * 100:.2f}% similarity.")

# Show the extracted shoes (optional)
cv2.imshow('Shoe 1', shoes[0])
cv2.imshow('Shoe 2', shoes[1])
cv2.waitKey(0)
cv2.destroyAllWindows()