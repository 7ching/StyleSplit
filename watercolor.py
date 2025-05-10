# watercolor effect
# reference: "Towards Photo Watercolorization with Artistic Verisimilitude"

from watercolor_utils import *

input_dir = "images"
input_filename = "train.jpg"
input_path = f"{input_dir}/{input_filename}"
output_path = f"{input_path.split('.')[0]}_output.jpg"

# load the image
image = cv2.imread(input_path)
resize_factor = image.shape[0] / 512
image = cv2.resize(
    image, (int(image.shape[1] / resize_factor), int(image.shape[0] / resize_factor))
)

image = adjust_color(image, model_path="./model/model.txt", style=-1)
saliency_map, dist_field = compute_saliency_distance_field(image)
segments, image = abstraction(image, dist_field, saliency_map)
boundary, grad_x, grad_y = boundary_classification(image, dist_field)
image = wet_in_wet(image, boundary, grad_x, grad_y, saliency_map, n_step=16)
image = hand_tremor(image, segments, boundary, get_perlin_noise)
image = edge_darkening(image)
image = granulation(image)
image = turbulence_flow(image, get_perlin_noise)
result = antialiasing(image)

# save the result
cv2.imwrite(output_path, result)
