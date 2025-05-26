import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

from segmentation import mask_generator, select_main
from oil_utils import process as process_oil
from watercolor_utils import (
    adjust_color,
    compute_saliency_distance_field,
    abstraction,
    boundary_classification,
    wet_in_wet,
    hand_tremor,
    edge_darkening,
    granulation,
    turbulence_flow,
    antialiasing,
    get_perlin_noise,
)
from comic_utils import process_comic_style
from sketch_utils import sketch as process_sketch

def apply_watercolor_style(image_np):
    """Applies the watercolor effect to an image."""
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    

    styled_image = adjust_color(image_bgr, model_path="./model/model.txt", style=-1)
    saliency_map, dist_field = compute_saliency_distance_field(styled_image)
    segments, styled_image = abstraction(styled_image, dist_field, saliency_map)
    boundary, grad_x, grad_y = boundary_classification(styled_image, dist_field)
    styled_image = wet_in_wet(styled_image, boundary, grad_x, grad_y, saliency_map, n_step=16)
    styled_image = hand_tremor(styled_image, segments, boundary, get_perlin_noise)
    styled_image = edge_darkening(styled_image)
    styled_image = granulation(styled_image)
    styled_image = turbulence_flow(styled_image, get_perlin_noise)
    result_bgr = antialiasing(styled_image)
    
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return result_rgb

st.title("StyleSplit: Mix Artistic Styles")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if "masks_data" not in st.session_state:
    st.session_state.masks_data = None
if "original_image_np" not in st.session_state:
    st.session_state.original_image_np = None
if "segmented_preview" not in st.session_state:
    st.session_state.segmented_preview = None


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    # resize h to 512
    if image.height > 512:
        resize_factor = image.height / 512
        new_width = int(image.width / resize_factor)
        image = image.resize((new_width, 512), Image.LANCZOS)
    original_image_np = np.array(image)
    st.session_state.original_image_np = original_image_np

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("1. Generate Masks (SAM)"):
        with st.spinner("Generating masks..."):
            masks = mask_generator(original_image_np)
            st.session_state.masks_data = masks
            
            preview_image = np.zeros_like(original_image_np)
            for i, mask_info in enumerate(masks):
                color = np.random.randint(50, 200, size=3)
                preview_image[mask_info["segmentation"]] = color
            
            alpha = 0.6
            blended_preview = cv2.addWeighted(original_image_np, 1 - alpha, preview_image, alpha, 0)
            st.session_state.segmented_preview = blended_preview
            st.success(f"Generated {len(masks)} masks!")

if st.session_state.segmented_preview is not None:
    st.image(st.session_state.segmented_preview, caption="Segmented Regions Preview", use_container_width=True)


if st.session_state.masks_data:
    st.subheader("2. Select Masks and Styles")
    
    num_masks = len(st.session_state.masks_data)
    selected_styles = {}

    # Create columns for better layout if many masks
    cols_per_row = 3
    
    for i in range(num_masks):
        # Display each mask individually for clarity
        mask_vis = np.zeros_like(st.session_state.original_image_np)
        mask_vis[st.session_state.masks_data[i]["segmentation"]] = [255, 255, 255] # White
        
        # Create a small preview of the mask area on the original image
        mask_preview = st.session_state.original_image_np.copy()
        mask_preview[~st.session_state.masks_data[i]["segmentation"]] = 0 # Black out non-mask area
        
        # Use columns for layout
        row_idx = i // cols_per_row
        col_idx = i % cols_per_row
        
        if col_idx == 0:
            cols = st.columns(cols_per_row)
        
        with cols[col_idx]:
            st.image(mask_preview, caption=f"Mask {i+1}", width=150)
            style = st.selectbox(
                f"Style for Mask {i+1}",
                ["None", "Oil Painting", "Watercolor", "Comic", "Sketch"],
                key=f"style_mask_{i}",
            )
            if style != "None":
                selected_styles[i] = style
    
    st.session_state.selected_styles = selected_styles

    if st.button("3. Apply Styles and Generate Output"):
        if not st.session_state.get("selected_styles"):
            st.warning("Please select a style for at least one mask.")
        else:
            with st.spinner("Applying styles..."):
                final_image = st.session_state.original_image_np.copy()
                
                for mask_idx, style_name in st.session_state.selected_styles.items():
                    mask_boolean = st.session_state.masks_data[mask_idx]["segmentation"]
                    
                    # Extract the region of interest based on the mask
                    # Create a full-sized image for styling, then mask it
                    image_to_style = st.session_state.original_image_np.copy()

                    styled_region_full = None

                    if style_name == "Oil Painting":
                        # Oil process might need BGR
                        image_bgr = cv2.cvtColor(image_to_style, cv2.COLOR_RGB2BGR)
                        # Ensure correct brushSize and expressionLevel, or make them configurable
                        styled_bgr = process_oil(image_bgr, brushSize=3, expressionLevel=2, seed=0)
                        styled_region_full = cv2.cvtColor(styled_bgr, cv2.COLOR_BGR2RGB)
                    elif style_name == "Watercolor":
                        styled_region_full = apply_watercolor_style(image_to_style)
                    elif style_name == "Comic":
                        # Comic process might need BGR
                        image_bgr = cv2.cvtColor(image_to_style, cv2.COLOR_RGB2BGR)
                        styled_bgr = process_comic_style(image_bgr) # Add params if needed
                        styled_region_full = cv2.cvtColor(styled_bgr, cv2.COLOR_BGR2RGB)
                    elif style_name == "Sketch":
                        # Sketch process might need BGR
                        image_bgr = cv2.cvtColor(image_to_style, cv2.COLOR_RGB2BGR)
                        styled_bgr = process_sketch(image_bgr)
                        styled_region_full = cv2.cvtColor(styled_bgr, cv2.COLOR_BGR2RGB)

                    if styled_region_full is not None:
                        # Ensure styled_region_full is same size as final_image
                        if styled_region_full.shape != final_image.shape:
                             styled_region_full = cv2.resize(styled_region_full, (final_image.shape[1], final_image.shape[0]))
                        
                        # Apply the styled region only where the mask is true
                        final_image[mask_boolean] = styled_region_full[mask_boolean]

                st.image(final_image, caption="Final Styled Image", use_container_width=True)
                
                # Provide download link
                result_image = Image.fromarray(final_image)
                buf = io.BytesIO()
                result_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="styled_image.png",
                    mime="image/png"
                )
                st.success("Image processed and ready for download!")

st.markdown("---")
st.markdown("Developed for StyleSplit project.")