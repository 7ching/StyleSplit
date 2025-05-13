import cv2
import numpy as np
import matplotlib.pyplot as plt

def sketch(img, paper_texture_path='./images/paper.jpg'):

    # 1. 邊緣保護平滑 (去雜訊，但保留大面積與輪廓)
    img_smooth = cv2.edgePreservingFilter(img, flags=1, sigma_s=20, sigma_r=0.2)

    # 2. 灰階
    gray = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY)

    # 3. 雙差分高斯 (DoG) 取得細節邊緣
    gauss1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.2)
    gauss2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    dog = cv2.subtract(gauss1, gauss2)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)

    # 4. 反轉 DoG 以得到線條 (黑線白底)
    edges_inv = 255 - dog

    # 5. 陰影層次 (反轉灰階)
    shade = gray.copy()

    # 6. 線條與陰影融合
    sketch_gray = cv2.addWeighted(edges_inv, 0.5, shade, 0.5, 0)
    sketch_gray = cv2.normalize(sketch_gray, None, 0, 255, cv2.NORM_MINMAX)

    # 7. 紙張紋理疊加 (增添手繪紙感)
    texture = cv2.imread(paper_texture_path, cv2.IMREAD_GRAYSCALE)
    if texture is not None:
        texture = cv2.resize(texture, (sketch_gray.shape[1], sketch_gray.shape[0]))
        sketch_gray = cv2.multiply(sketch_gray, texture, scale=1/255)

    # 8. 轉回 BGR
    sketch_bgr = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)
    return sketch_bgr
