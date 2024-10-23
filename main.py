import cv2
import numpy as np

def extract_color_difference(image_path, roi_new, roi_old):
    # Load the image
    image = cv2.imread(image_path)
    
    # Extract the ROIs
    roi_new_img = image[roi_new[1]:roi_new[1]+roi_new[3], roi_new[0]:roi_new[0]+roi_new[2]]
    roi_old_img = image[roi_old[1]:roi_old[1]+roi_old[3], roi_old[0]:roi_old[0]+roi_old[2]]
    
    # Convert ROIs to HSV color space
    roi_new_hsv = cv2.cvtColor(roi_new_img, cv2.COLOR_BGR2HSV)
    roi_old_hsv = cv2.cvtColor(roi_old_img, cv2.COLOR_BGR2HSV)
    
    # Calculate the mean color of each ROI
    mean_new = cv2.mean(roi_new_hsv)[:3]
    mean_old = cv2.mean(roi_old_hsv)[:3]
    
    # Calculate the difference in color
    color_difference = np.abs(np.array(mean_new) - np.array(mean_old))
    
    return color_difference

# Example usage
image_path = 'shoes.jpg'
roi_new = (x1, y1, width1, height1)  # Replace with actual coordinates
roi_old = (x2, y2, width2, height2)  # Replace with actual coordinates

color_diff = extract_color_difference(image_path, roi_new, roi_old)
print("Color difference (H, S, V):", color_diff)