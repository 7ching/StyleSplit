# StyleSplit: Mix Artistic Styles on Images

## Overview

StyleSplit is an interactive image styling application built with Streamlit. It allows users to upload images, automatically segment them into regions using SAM (Segment Anything Model), and apply different artistic styles to each region. Supported styles include oil painting, watercolor, comic, and sketch. Users can preview intermediate results, blend styles across regions, and download the final styled image.

## Features

- **Automatic Mask Generation**: Generate segmentation masks for uploaded images using a pre-trained SAM model.
- **Multiple Artistic Styles**: Apply oil painting, watercolor, comic, and sketch effects.
- **Region-based Styling**: Mix and match styles on individual image regions.
- **Live Preview**: View segmented regions and styled results in real time.
- **One-click Download**: Download the final styled image as a PNG file.

## Directory Structure

```text
StyleSplit/
├── app.py                   # Main Streamlit app
├── segmentation.py          # Mask generation utilities (SAM wrapper)
├── comic_utils.py           # Comic style processing
├── oil_utils.py             # Oil painting style processing
├── sketch_utils.py          # Sketch style processing
├── watercolor_utils.py      # Watercolor style processing
├── model/
│   ├── model.txt            # Pre-trained parameters for watercolor effect
│   └── readme.md            # Details about the model parameters
├── ckpt/
│   └── sam_vit_h_4b8939.pth # SAM model checkpoint
├── images/                  # Sample input and output images
├── output/                  # Generated composite outputs
└── README.md                # Project overview and instructions
``` 

## Dependencies

- Python 3.11
- Streamlit
- Pillow
- NumPy
- OpenCV (`opencv-python`)
- PyTorch (with matching CUDA/cuDNN if using GPU)
- Segment Anything
 
You can install the necessary packages via pip:

```bash
pip install segment_anything streamlit pillow numpy opencv-python torch torchvision 
```

## Setup

1. **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/StyleSplit.git
    cd StyleSplit
    ```

2. **Download the SAM checkpoint**

   Place `sam_vit_h_4b8939.pth` into the `ckpt/` directory. You can download it from the [Link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

## Usage

1. **Run the Streamlit app**

   ```bash
    streamlit run app.py
    ```

2. **Upload an image**

   Click **Browse files** to upload a JPG or PNG image.

3. **Generate Masks**

   Press **1. Generate Masks (SAM)** to segment the image into regions.

4. **Select Styles**

   Under **2. Select Masks and Styles**, choose a style for each segmented region and click **Apply Styles**.

5. **Preview and Download**

   - View the live preview of styled regions and the final composite image.
   - Click **Download Image** to save the result as a PNG file.
  
## Team Members
- **R13944008 吳宜宸**
- **R13922A04 操之晴**
- **R13944035 張翔**