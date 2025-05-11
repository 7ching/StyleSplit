import random
import time
import numpy as np
from skimage import draw
from skimage.color import rgb2gray
from skimage.filters import laplace as skimage_laplace # 避免名稱衝突
from scipy.ndimage import sobel
from math import pi

BRUSHES = 50

def process(inputImage, brushSize, expressionLevel, brushes=BRUSHES, seed=None, 
            color_jitter_strength=0,
            stroke_spacing_factor=0.7,
            gradient_align_strength=0.0,
            adaptive_brush_enabled=False,
            adaptive_brush_strength=0.5,
            min_brush_scale=0.5,
            max_brush_scale=1.2):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 設定筆刷主半徑
    base_r_radius = float(brushSize)
    if base_r_radius < 1.0:
        base_r_radius = 1.0

    # 前處理：灰階、梯度、細節圖
    gray_image = None
    grad_orientation = None
    detail_map_norm = None

    if gradient_align_strength > 0 or adaptive_brush_enabled:
        if inputImage.ndim == 3 and inputImage.shape[2] == 3:
            gray_image = rgb2gray(inputImage)
        elif inputImage.ndim == 2:
            gray_image = inputImage
        else:
            gray_image = rgb2gray(inputImage[:,:,:3]) if inputImage.shape[2] > 2 else rgb2gray(inputImage)

    if gradient_align_strength > 0 and gray_image is not None:
        gx = sobel(gray_image, axis=1) # y 方向梯度
        gy = sobel(gray_image, axis=0) # x 方向梯度
        grad_orientation = np.arctan2(gy, gx) # 梯度角度

    if adaptive_brush_enabled and gray_image is not None:
        detail_map = np.abs(skimage_laplace(gray_image))
        min_detail, max_detail = np.min(detail_map), np.max(detail_map)
        if max_detail - min_detail < 1e-6:
            detail_map_norm = np.zeros_like(detail_map)
        else:
            detail_map_norm = (detail_map - min_detail) / (max_detail - min_detail)

    # 預先產生筆刷模板
    actual_expression_level = max(0.1, expressionLevel)

    brush_templates = []
    for _ in range(brushes):
        min_c_factor = min(1.0, actual_expression_level)
        max_c_factor = max(1.0, actual_expression_level)
        c_radius_val = base_r_radius * random.uniform(min_c_factor, max_c_factor)
        random_rot = random.random() * pi
        brush_templates.append({'c_radius_val_at_base': c_radius_val, 'random_rotation': random_rot})

    # 計算邊界
    max_possible_r_radius = base_r_radius * max_brush_scale
    max_possible_c_radius_factor = max(1.0, actual_expression_level)
    max_abs_radius_component = max_possible_r_radius * max_possible_c_radius_factor
    margin = int(np.ceil(max_abs_radius_component)) + 5

    result = np.zeros(inputImage.shape, dtype=np.uint8)
    
    # 筆刷間距
    step_size = max(1, int(base_r_radius * 2 * stroke_spacing_factor))

    for x in range(margin, inputImage.shape[0] - margin, step_size):
        for y in range(margin, inputImage.shape[1] - margin, step_size):
            # 1. 決定目前筆刷主半徑
            current_r_radius = base_r_radius
            if adaptive_brush_enabled and detail_map_norm is not None:
                detail_val = detail_map_norm[x, y]
                target_scale = min_brush_scale + detail_val * (max_brush_scale - min_brush_scale)
                current_scale = (1.0 - adaptive_brush_strength) * 1.0 + adaptive_brush_strength * target_scale
                current_r_radius = base_r_radius * current_scale
            current_r_radius = max(1.0, current_r_radius)

            # 2. 選擇模板並決定副半徑
            template = random.choice(brush_templates)
            current_c_radius = template['c_radius_val_at_base'] * (current_r_radius / base_r_radius)
            current_c_radius = max(1.0, current_c_radius)

            # 3. 計算旋轉角度
            final_rotation = template['random_rotation']
            if gradient_align_strength > 0 and grad_orientation is not None:
                target_grad_rotation = grad_orientation[x, y] + pi / 2.0
                final_rotation = (1.0 - gradient_align_strength) * template['random_rotation'] + \
                                 gradient_align_strength * target_grad_rotation
                final_rotation = final_rotation % (2 * pi)

            # 4. 畫橢圓
            canvas_dim_approx = int(np.ceil(max(current_r_radius, current_c_radius) * 2.0)) + 4
            canvas_center = canvas_dim_approx // 2
            try:
                rr, cc = draw.ellipse(r=canvas_center, c=canvas_center,
                                      r_radius=current_r_radius, c_radius=current_c_radius,
                                      rotation=final_rotation, shape=(canvas_dim_approx, canvas_dim_approx))
            except Exception as e:
                continue

            # 5. 取得 final color
            base_color = inputImage[x, y].astype(np.float32)
            if color_jitter_strength > 0:
                jitter = np.random.randint(-color_jitter_strength, color_jitter_strength + 1, size=base_color.shape)
                jittered_color = base_color + jitter
                final_color = np.clip(jittered_color, 0, 255).astype(np.uint8)
            else:
                final_color = base_color.astype(np.uint8)
            
            # 6. 將筆刷畫到結果圖
            target_x_coords = x + rr - canvas_center
            target_y_coords = y + cc - canvas_center
            valid_mask = (target_x_coords >= 0) & (target_x_coords < result.shape[0]) & \
                         (target_y_coords >= 0) & (target_y_coords < result.shape[1])
            valid_target_x = target_x_coords[valid_mask]
            valid_target_y = target_y_coords[valid_mask]
            if valid_target_x.size > 0 and valid_target_y.size > 0 :
                 result[valid_target_x, valid_target_y] = final_color
                 
    return result
