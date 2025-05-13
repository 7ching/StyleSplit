import cv2
import numpy as np

def process_comic_style(image, canny_thresh1=50, canny_thresh2=150, num_colors_kmeans=8, gaussian_blur_ksize=(5,5)):
    if image is None:
        print("Error: Input image is None")
        return None

    # 1. Edge Detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, gaussian_blur_ksize, 0)
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    
    # 2. Color Simplification/Quantization
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    compactness, labels, centers = cv2.kmeans(pixels, num_colors_kmeans, None, criteria, 10, flags)
    
    centers = np.uint8(centers)
    quantized_img_flat = centers[labels.flatten()]
    quantized_img = quantized_img_flat.reshape(image.shape)

    # 3. Combine Edges and Color
    comic_style_image = quantized_img.copy()
    comic_style_image[edges == 255] = [0, 0, 0]

    return comic_style_image