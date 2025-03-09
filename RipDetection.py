import cv2
import numpy as np

class EdgeDetection:
    def __init__(self, image, window_name, screen_width, screen_height):
        self.image = image  # Original image
        self.window_name = window_name
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Resize image to fit screen size
        self.resized_image = cv2.resize(self.image, (self.screen_width, self.screen_height))

    def detect_edges(self):
        gray_image = cv2.cvtColor(self.resized_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 150, 250)

        ## Create an empty image to store the colored edges
        color_edges = np.zeros_like(self.resized_image)

        ## Define the area which is considered  the top shoe and bottom shoe
        top_shoe_coords = (0, 0, self.screen_width, int(self.screen_height * 0.45)) 
        bottom_shoe_coords = (0, int(self.screen_height * 0.45), self.screen_width, self.screen_height)  

        ## Create masks for the top and bottom shoe areas
        top_shoe_mask = np.zeros_like(edges)
        top_shoe_mask[top_shoe_coords[1]:top_shoe_coords[3], top_shoe_coords[0]:top_shoe_coords[2]] = edges[top_shoe_coords[1]:top_shoe_coords[3], top_shoe_coords[0]:top_shoe_coords[2]]

        bottom_shoe_mask = np.zeros_like(edges)
        bottom_shoe_mask[bottom_shoe_coords[1]:bottom_shoe_coords[3], bottom_shoe_coords[0]:bottom_shoe_coords[2]] = edges[bottom_shoe_coords[1]:bottom_shoe_coords[3], bottom_shoe_coords[0]:bottom_shoe_coords[2]]

        ## Apply blue to the top shoe edges
        color_edges[top_shoe_mask != 0] = [255, 0, 0]  # Blue in BGR format
        
        ## Apply red to the bottom shoe edges
        color_edges[bottom_shoe_mask != 0] = [0, 0, 255]  # Red in BGR format

        ##Count the number of non-zero edge pixels in the blue (top) and red (bottom) areas
        top_shoe_edge_count = np.count_nonzero(top_shoe_mask)
        bottom_shoe_edge_count = np.count_nonzero(bottom_shoe_mask)

        # Display the result with edge counting information
        print(f"Top shoe edge count (blue): {top_shoe_edge_count}")
        print(f"Bottom shoe edge count (red): {bottom_shoe_edge_count}")

        # Determine which shoe has more edges/rips
        if top_shoe_edge_count > bottom_shoe_edge_count:
            result_text = "The top shoe has more edges/rips."
        elif bottom_shoe_edge_count > top_shoe_edge_count:
            result_text = "The bottom shoe has more edges/rips."
        else:
            result_text = "Both shoes have an equal number of edges/rips."


        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5 * (self.screen_width / 800)
        font_thickness = 1 
        text_color = (255, 255, 255) 

  
        text_x = 10
        text_y = 30

        # Overlay the dynamic text on the image
        cv2.putText(color_edges, f"Top shoe edge count: {top_shoe_edge_count}", (text_x, text_y),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        cv2.putText(color_edges, f"Bottom shoe edge count: {bottom_shoe_edge_count}", (text_x, text_y + 30),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        cv2.putText(color_edges, result_text, (text_x, text_y + 60),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Display the edge-detected image with colored edges and text
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(self.window_name, color_edges)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Define screen dimensions dynamically
screen_width = 800
screen_height = 600

image = cv2.imread("./TwoShoes.jpg")

window_name = "Edge Detection"
edge_detector = EdgeDetection(image, window_name, screen_width, screen_height)
edge_detector.detect_edges()
