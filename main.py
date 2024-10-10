import cv2
import numpy as np

## loading background image
background_image = cv2.imread("background.jpg") 
background_image = cv2.resize(background_image, (1280, 720))


tint = np.zeros_like(background_image, dtype=np.uint8)
tint[:] = (0, 0, 0)  ## Solid black
alpha = 0.9  ## This controls tint strength, alpha of the solid black tint array above in line 10
tinted_background = cv2.addWeighted(background_image, 1 - alpha, tint, alpha, 0) ## applying tint


# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 450 <= x <= 830 and 300 <= y <= 350:  # Button 1 area
            print("Option 1 selected")
        elif 450 <= x <= 830 and 370 <= y <= 420:  # Button 2 area
            print("Option 2 selected")
        elif 450 <= x <= 830 and 440 <= y <= 490:  # Button 3 area
            print("Option 3 selected")

# Set the window name and create a fullscreen window
cv2.namedWindow("Wear and Tear Shoe Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Wear and Tear Shoe Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Centered positioning
title_text = "Wear and Tear Shoe Detection"
prompt_text = "Please select an option"
button1_text = "Option 1"
button2_text = "Option 2"
button3_text = "Option 3"

# Set fonts and sizes
font_title = cv2.FONT_HERSHEY_COMPLEX
font_prompt = cv2.FONT_HERSHEY_SIMPLEX
font_buttons = cv2.FONT_HERSHEY_SIMPLEX
font_scale_title = 2
font_scale_prompt = 1
font_scale_buttons = 0.8
font_thickness = 3

# Calculate text sizes
(title_width, title_height), baseline_title = cv2.getTextSize(title_text, font_title, font_scale_title, font_thickness)
(prompt_width, prompt_height), baseline_prompt = cv2.getTextSize(prompt_text, font_prompt, font_scale_prompt, font_thickness)
(button1_width, button1_height), baseline_button1 = cv2.getTextSize(button1_text, font_buttons, font_scale_buttons, font_thickness)
(button2_width, button2_height), baseline_button2 = cv2.getTextSize(button2_text, font_buttons, font_scale_buttons, font_thickness)
(button3_width, button3_height), baseline_button3 = cv2.getTextSize(button3_text, font_buttons, font_scale_buttons, font_thickness)

# Draw the tinted background
image = tinted_background.copy()

# Draw the GUI elements centered
cv2.putText(image, title_text, ((1280 - title_width) // 2, 100), font_title, font_scale_title, (255, 255, 255), font_thickness)
cv2.putText(image, prompt_text, ((1280 - prompt_width) // 2, 200), font_prompt, font_scale_prompt, (255, 255, 255), 2)

# Draw buttons with black background and white outline
button_height = 50
button_coords = [(450, 300, 830, 350), (450, 370, 830, 420), (450, 440, 830, 490)]  # (x1, y1, x2, y2)

for (x1, y1, x2, y2) in button_coords:
    # Draw black button background
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)  # Black background
    # Draw white outline
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # White outline

# Add button labels in white
cv2.putText(image, button1_text, ((1280 - button1_width) // 2, 335), font_buttons, font_scale_buttons, (255, 255, 255), 2)
cv2.putText(image, button2_text, ((1280 - button2_width) // 2, 405), font_buttons, font_scale_buttons, (255, 255, 255), 2)
cv2.putText(image, button3_text, ((1280 - button3_width) // 2, 475), font_buttons, font_scale_buttons, (255, 255, 255), 2)

# Set the mouse callback function
cv2.setMouseCallback("Wear and Tear Shoe Detection", mouse_callback)

while True:
    # Show the image
    cv2.imshow("Wear and Tear Shoe Detection", image)

    # Check for key events
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Exit on space bar
        break

# Clean up
cv2.destroyAllWindows()
