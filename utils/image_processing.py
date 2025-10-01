import cv2
import numpy as np
from PIL import Image

def validate_image(image_path, min_std_dev=12, min_dims=(200, 200)):
    """
    Validates an image to see if it's suitable for crack detection.
    Rejection condition: The image will be rejected if it's too small, not a valid image,
    or if it's too uniform (e.g., a solid color), which is determined by a low standard deviation
    of pixel intensities.
    """
    try:
        img = Image.open(image_path)
        img_np = np.array(img)
    except Exception:
        return False, "Invalid or corrupt image file."

    if img_np.ndim < 2:
        return False, "Image format is not supported."

    if img_np.shape[0] < min_dims[0] or img_np.shape[1] < min_dims[1]:
        return False, f"Image is too small. Please upload an image with minimum dimensions of {min_dims[0]}x{min_dims[1]} pixels."

    if img_np.ndim == 3:
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img_np
        
    std_dev = np.std(gray_img)
    if std_dev < min_std_dev:
        return False, f"Image lacks sufficient detail or contrast (Std Dev: {std_dev:.2f}). It may be a uniform surface or out of focus."

    return True, "Image is valid."


def advanced_crack_refinement(pred_mask, min_area=25, close_kernel_size=5):
    """
    Cleans the raw binary prediction mask by removing small noise and closing small gaps.
    This function is copied directly from your training script.
    """
    mask = pred_mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 1

    if close_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
    return cleaned_mask