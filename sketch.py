from sketch_utils import *
import cv2

input_dir = "images"
input_filenames = ["cat.jpg", "house.jpg", "train.jpg"]

for input_filename in input_filenames:
    input_path = f"{input_dir}/{input_filename}"
    output_path = f"{input_path.split('.')[0]}_output_sketch.jpg"

    image = cv2.imread(input_path)

    result = sketch(image)
    
    cv2.imwrite(output_path, result)