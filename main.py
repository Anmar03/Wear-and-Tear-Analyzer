import cv2
import numpy as np
from screeninfo import get_monitors  # Need this to support getting screen size for MacOS devices
import easygui
from DeformationDetector import DeformationDetector #Máté's class

exit_program = False
monitor = get_monitors()[0]  # Gets the screen size of the primary monitor
screen_width = monitor.width
screen_height = monitor.height

# Resize background image to screen size
background_image = cv2.imread("background.jpg")
background_image = cv2.resize(background_image, (screen_width, screen_height))

# Set texts for the buttons on the UI
title_text = "Wear and Tear Shoe Detection"
prompt_text = "Please select one of the following:"
button1_text = "Analyze Shoes"
button2_text = "Upload Image"
button3_text = "Exit"

# Fonts
font_title = cv2.FONT_HERSHEY_COMPLEX
font_prompt = cv2.FONT_HERSHEY_SIMPLEX
font_buttons = cv2.FONT_HERSHEY_SIMPLEX
font_scale_title = 2
font_scale_prompt = 1
font_scale_buttons = 0.8
font_thickness = 3

# Initialize imageToAnalyze to None (no default image) (forces user to upload an image)
imageToAnalyze = None
detector = DeformationDetector(imageToAnalyze, "Wear and Tear Shoe Detection", screen_width, screen_height)

# Create a solid black tint over the background image
tint = np.zeros_like(background_image, dtype=np.uint8)
tint[:] = (0, 0, 0)  # Solid black
alpha = 0.7  # Adjusted tint strength to make the background slightly brighter
tinted_background = cv2.addWeighted(background_image, 1 - alpha, tint, alpha, 0)  # Apply tint

# Redraw the main menu
def draw_main_menu():
    # Draw the tinted background to the screen
    global image
    image = tinted_background.copy()

    # Draw the buttons on the screen
    button_coords = [button1_coords, button2_coords, button3_coords]
    for (x1, y1, x2, y2) in button_coords:
        # Transparent button background
        button_bg = image[y1:y2, x1:x2]
        overlay = button_bg.copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.6, button_bg, 0.4, 0, button_bg)

        # Draw button borders
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Draw text on the screen centered
    cv2.putText(image, title_text, ((screen_width - title_width) // 2, 150), font_title, font_scale_title,
                (255, 255, 255), font_thickness)
    cv2.putText(image, prompt_text, ((screen_width - prompt_width) // 2, 250), font_prompt, font_scale_prompt,
                (255, 255, 255), 2)
    cv2.putText(image, button1_text,
                ((screen_width - button1_width) // 2, button1_coords[1] + button_height // 2 + 10), font_buttons,
                font_scale_buttons, (255, 255, 255), 2)
    cv2.putText(image, button2_text,
                ((screen_width - button2_width) // 2, button2_coords[1] + button_height // 2 + 10), font_buttons,
                font_scale_buttons, (255, 255, 255), 2)
    cv2.putText(image, button3_text,
                ((screen_width - button3_width) // 2, button3_coords[1] + button_height // 2 + 10), font_buttons,
                font_scale_buttons, (255, 255, 255), 2)

    # Show the main menu
    cv2.imshow("Wear and Tear Shoe Detection", image)

    # **Reset the mouse callback to the main menu**
    cv2.setMouseCallback("Wear and Tear Shoe Detection", mouse_callback)

# Callback function for mouse 
def mouse_callback(event, x, y, flags, param):
    global exit_program, imageToAnalyze, detector
    if event == cv2.EVENT_LBUTTONDOWN: # click left button on mouse
        if button1_coords[0] <= x <= button1_coords[2] and button1_coords[1] <= y <= button1_coords[3]:  # Analyze Shoes
            if detector.image is not None:
                detector.clear_screen() #Background black
                detector.run() # Show the actual analyzing
                # After analysis, redraw the main menu
                draw_main_menu()
            else:
                # Show a warning if no image has been uploaded
                warning_image = image.copy()
                warning_text = "Please upload an image first."
                (warning_width, warning_height), _ = cv2.getTextSize(warning_text, font_prompt, 1.2, 2)
                cv2.putText(warning_image, warning_text, ((screen_width - warning_width) // 2, screen_height - 200),
                            font_prompt, 1.2, (0, 0, 255), 2)
                cv2.imshow("Wear and Tear Shoe Detection", warning_image)
                cv2.waitKey(2000)  # Display the warning for 2 seconds
                draw_main_menu() # Go back to main menu after warning is gone
        elif button2_coords[0] <= x <= button2_coords[2] and button2_coords[1] <= y <= button2_coords[3]:  # Upload Image
            image_path = easygui.fileopenbox(filetypes=["*.jpg", "*.jpeg", "*.png"]) #Types of files allowed
            if image_path:
                imageToAnalyze = cv2.imread(image_path)
                if imageToAnalyze is not None:
                    detector.set_image(imageToAnalyze)
                    print("Image uploaded successfully.")
                    # Show a confirmation message on the main menu
                    confirmation_image = image.copy()
                    confirmation_text = "Image uploaded successfully!" #Display on menu
                    (confirmation_width, confirmation_height), _ = cv2.getTextSize(confirmation_text, font_prompt, 1.2, 2)
                    cv2.putText(confirmation_image, confirmation_text, ((screen_width - confirmation_width) // 2, screen_height - 200),
                                font_prompt, 1.2, (0, 255, 0), 2)
                    cv2.imshow("Wear and Tear Shoe Detection", confirmation_image)
                    cv2.waitKey(1500)  # Display the confirmation for 1.5 seconds
                    draw_main_menu()
                else:
                    print("Failed to load the image.")
            else:
                print("No image selected.")
        elif button3_coords[0] <= x <= button3_coords[2] and button3_coords[1] <= y <= button3_coords[3]:  # Exit
            exit_program = True  # Program will  exit

# Create the fullscreen window
cv2.namedWindow("Wear and Tear Shoe Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Wear and Tear Shoe Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Get text sizes for dynamic button dimensions
(title_width, title_height), _ = cv2.getTextSize(title_text, font_title, font_scale_title, font_thickness)
(prompt_width, prompt_height), _ = cv2.getTextSize(prompt_text, font_prompt, font_scale_prompt, font_thickness)
(button1_width, button1_height), _ = cv2.getTextSize(button1_text, font_buttons, font_scale_buttons, font_thickness)
(button2_width, button2_height), _ = cv2.getTextSize(button2_text, font_buttons, font_scale_buttons, font_thickness)
(button3_width, button3_height), _ = cv2.getTextSize(button3_text, font_buttons, font_scale_buttons, font_thickness)

# Scale buttons based on screen height
button_height = int(0.07 * screen_height)  # Button height is 7% of the screen height
button_width = int(0.3 * screen_width)     # Button width is 30% of screen width
button_start = (screen_width - button_width) // 2  # Centered button position

# Screen coordinates for buttons
button1_coords = (button_start, 350, button_start + button_width, 350 + button_height)
button2_coords = (button_start, 450, button_start + button_width, 450 + button_height)
button3_coords = (button_start, 550, button_start + button_width, 550 + button_height)

# Draw the main menu
draw_main_menu()

# Set the mouse callback function
cv2.setMouseCallback("Wear and Tear Shoe Detection", mouse_callback)

# Main program loop
while True:
    if exit_program:
        cv2.destroyAllWindows()
        break
    cv2.waitKey(1)  # Process mouse clicks
