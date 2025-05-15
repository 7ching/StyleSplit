from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np


def mask_generator(image, ckpt=r"./ckpt/sam_vit_h_4b8939.pth"):
    sam = sam_model_registry["default"](checkpoint=ckpt) # download from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=5000)
    masks = mask_generator.generate(image)

    H, W = image.shape[:2]
    area_threshold = 0.05 * H * W 
    filtered_masks = [
        m for m in masks if np.sum(m["segmentation"]) >= area_threshold
    ]

    masks_image = np.zeros(image.shape, dtype=np.uint8)
    for mask in filtered_masks:
        masks_image[mask["segmentation"]] = np.random.randint(0, 255, size=3)

    return filtered_masks

def select_main(filtered_masks, idx):
    return filtered_masks[idx]["segmentation"]


# usage example
# masks = mask_generator(image)
# plot to check
# selected_mask = select_main(masks, 0)