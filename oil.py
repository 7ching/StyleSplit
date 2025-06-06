from oil_utils import process
import cv2

input_dir = "images"
input_filenames = ["cat.jpg", "house.jpg", "train.jpg"]

for input_filename in input_filenames:
    input_path = f"{input_dir}/{input_filename}"
    output_path = f"{input_path.split('.')[0]}_output_oil.jpg"

    image = cv2.imread(input_path)
    resize_factor = image.shape[0] / 512
    image = cv2.resize(
        image, (int(image.shape[1] / resize_factor), int(image.shape[0] / resize_factor))
    )


    result = process(image, brushSize=3, expressionLevel=2, seed=0)

    cv2.imwrite(output_path, result)