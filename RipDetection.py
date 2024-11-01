import cv2
import numpy as np

class RipDetection:
    def __init__(self, image, window_name, screen_width, screen_height):
        self.image = image
        self.window_name = window_name
        self.screen_width = screen_width
        self.screen_height = screen_height

    def clear_screen(self):
        """Clears the screen by displaying a black image."""
        black_image = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        cv2.imshow(self.window_name, black_image)

    def detect_rips(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)


        edges = cv2.Canny(blurred_image, 150, 250)

        red_overlay = np.zeros_like(self.image)
        red_overlay[edges != 0] = [0, 0, 255]  


        overlay = cv2.addWeighted(self.image, 0.7, red_overlay, 0.3, 0)


        cv2.imshow(self.window_name, overlay)

        self.crop_edges(edges)

    def crop_edges(self, edges):

        top_shoe_coords = (50, 50, 700, 350)  
        bottom_shoe_coords = (50, 350, 700, 650)  

 
        top_shoe_edges = edges[top_shoe_coords[1]:top_shoe_coords[3], top_shoe_coords[0]:top_shoe_coords[2]]
        bottom_shoe_edges = edges[bottom_shoe_coords[1]:bottom_shoe_coords[3], bottom_shoe_coords[0]:bottom_shoe_coords[2]]


        top_shoe_edge_image = np.zeros((top_shoe_edges.shape[0], top_shoe_edges.shape[1], 3), dtype=np.uint8)
        top_shoe_edge_image[top_shoe_edges != 0] = [0, 0, 255]  

        bottom_shoe_edge_image = np.zeros((bottom_shoe_edges.shape[0], bottom_shoe_edges.shape[1], 3), dtype=np.uint8)
        bottom_shoe_edge_image[bottom_shoe_edges != 0] = [0, 0, 255] 


        cv2.imwrite("top_shoe_edges.jpg", top_shoe_edge_image)
        cv2.imwrite("bottom_shoe_edges.jpg", bottom_shoe_edge_image)

        cv2.imshow("Top Shoe Edges", top_shoe_edge_image)
        cv2.imshow("Bottom Shoe Edges", bottom_shoe_edge_image)

    def run(self):
        self.detect_rips()
        cv2.waitKey(0) 
        cv2.destroyAllWindows()


image = cv2.imread("./TwoShoes.jpg")
image = cv2.resize(image, (800, 600))


screen_width, screen_height = 800, 600

window_name = "Wear and Tear Shoe Detection"
rip_detector = RipDetection(image, window_name, screen_width, screen_height)

rip_detector.run()
