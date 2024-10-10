import cv2
import numpy as np
from screeninfo import get_monitors

# Retrieve the current screen size
monitor = get_monitors()[0]  # Get the first monitor (primary)
screen_width = monitor.width
screen_height = monitor.height

# Load the background image and resize it
background_image = cv2.imread("background.jpg")
background_image = cv2.resize(background_image, (screen_width, screen_height))

# Create a solid black overlay tint
tint = np.zeros_like(background_image, dtype=np.uint8)
tint[:] = (0, 0, 0)  # Solid black
alpha = 0.8  # Tint strength
tinted_background = cv2.addWeighted(background_image, 1 - alpha, tint, alpha, 0)  # Applying tint

# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if button1_coords[0] <= x <= button1_coords[2] and button1_coords[1] <= y <= button1_coords[3]:  # Button 1 area
            print("Option 1 selected")
        elif button2_coords[0] <= x <= button2_coords[2] and button2_coords[1] <= y <= button2_coords[3]:  # Button 2 area
            print("Option 2 selected")
        elif button3_coords[0] <= x <= button3_coords[2] and button3_coords[1] <= y <= button3_coords[3]:  # Button 3 area
            print("Option 3 selected")

# Create a fullscreen window
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
(title_width, title_height), _ = cv2.getTextSize(title_text, font_title, font_scale_title, font_thickness)
(prompt_width, prompt_height), _ = cv2.getTextSize(prompt_text, font_prompt, font_scale_prompt, font_thickness)
(button1_width, button1_height), _ = cv2.getTextSize(button1_text, font_buttons, font_scale_buttons, font_thickness)
(button2_width, button2_height), _ = cv2.getTextSize(button2_text, font_buttons, font_scale_buttons, font_thickness)
(button3_width, button3_height), _ = cv2.getTextSize(button3_text, font_buttons, font_scale_buttons, font_thickness)

# Draw the tinted background
image = tinted_background.copy()

# Draw the GUI elements centered
cv2.putText(image, title_text, ((screen_width - title_width) // 2, 100), font_title, font_scale_title, (255, 255, 255), font_thickness)
cv2.putText(image, prompt_text, ((screen_width - prompt_width) // 2, 200), font_prompt, font_scale_prompt, (255, 255, 255), 2)

# Define button dimensions (scaled based on screen height)
button_height = int(0.07 * screen_height)  # Scaled height for buttons
button_width = int(0.3 * screen_width)     # Scaled width for buttons
x_start = (screen_width - button_width) // 2  # Centered button position

# Coordinates for buttons (scaled)
button1_coords = (x_start, 300, x_start + button_width, 300 + button_height)
button2_coords = (x_start, 370, x_start + button_width, 370 + button_height)
button3_coords = (x_start, 440, x_start + button_width, 440 + button_height)

# Draw buttons with black background and white outline
button_coords = [button1_coords, button2_coords, button3_coords]
for (x1, y1, x2, y2) in button_coords:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)  # Black background
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # White outline

# Add button labels in white (centered based on scaled button coordinates)
cv2.putText(image, button1_text, ((screen_width - button1_width) // 2, button1_coords[1] + button_height // 2 + 10), font_buttons, font_scale_buttons, (255, 255, 255), 2)
cv2.putText(image, button2_text, ((screen_width - button2_width) // 2, button2_coords[1] + button_height // 2 + 10), font_buttons, font_scale_buttons, (255, 255, 255), 2)
cv2.putText(image, button3_text, ((screen_width - button3_width) // 2, button3_coords[1] + button_height // 2 + 10), font_buttons, font_scale_buttons, (255, 255, 255), 2)

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
