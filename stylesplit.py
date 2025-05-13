from sketch_utils import *
from oil_utils import *
from watercolor_utils import *
import cv2

# input_dir = "images"
# input_filenames = ["cat.jpg", "house.jpg", "train.jpg"]

input_dir = "images"
output_dir = "output"
input_filename = "cat.jpg"
input_path = f"{input_dir}/{input_filename}"
output_path = f"{output_dir}/{input_filename.split('.')[0]}"

# load the image
image = cv2.imread(input_path)
resize_factor = image.shape[0] / 512
resize_image = cv2.resize(
    image, (int(image.shape[1] / resize_factor), int(image.shape[0] / resize_factor))
)
image_rgb = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)

# sketch effect
sketch_result = sketch(resize_image)

# oil painting effect
oil_result = process(image_rgb, brushSize=1.5, expressionLevel=2, seed=0)
oil_result_bgr = cv2.cvtColor(oil_result, cv2.COLOR_RGB2BGR)

# watercolor effect
water_img = adjust_color(resize_image, model_path="./model/model.txt", style=-1)
saliency_map, dist_field = compute_saliency_distance_field(water_img)
segments, water_img = abstraction(water_img, dist_field, saliency_map)
boundary, grad_x, grad_y = boundary_classification(water_img, dist_field)
water_img = wet_in_wet(water_img, boundary, grad_x, grad_y, saliency_map, n_step=16)
water_img = hand_tremor(water_img, segments, boundary, get_perlin_noise)
water_img = edge_darkening(water_img)
water_img = granulation(water_img)
water_img = turbulence_flow(water_img, get_perlin_noise)
water_result = antialiasing(water_img)

# 使用 GrabCut 自動分割前景（主體）和背景
mask = np.zeros(resize_image.shape[:2], np.uint8)
rect = (10, 10, resize_image.shape[1] - 20, resize_image.shape[0] - 20)  # 初始化矩形：去除邊界
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv2.grabCut(resize_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

combined1 = np.where(mask2[:, :, None] == 1, sketch_result, water_result)
combined2 = np.where(mask2[:, :, None] == 1, water_result, oil_result_bgr)
combined3 = np.where(mask2[:, :, None] == 1, oil_result_bgr, sketch_result)


cv2.imwrite(f"{output_path}_s+w.jpg", combined1)
cv2.imwrite(f"{output_path}_w+o.jpg", combined2)
cv2.imwrite(f"{output_path}_o+s.jpg", combined3)

