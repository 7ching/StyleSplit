from comic_utils import process_comic_style
import cv2

input_dir = "images"
input_filenames = ["cat.jpg", "house.jpg", "train.jpg"]

canny_threshold1 = 50
canny_threshold2 = 150
num_colors = 8
gaussian_ksize = (5,5)

for input_filename in input_filenames:
    input_path = f"{input_dir}/{input_filename}"
    output_path = f"{input_path.split('.')[0]}_output_comic.jpg"

    image = cv2.imread(input_path)

    result = process_comic_style(
        image,
        canny_thresh1=canny_threshold1,
        canny_thresh2=canny_threshold2,
        num_colors_kmeans=num_colors,
        gaussian_blur_ksize=gaussian_ksize
    )

    cv2.imwrite(output_path, result)