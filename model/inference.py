import torch
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model.architecture import EnhancedCrackCapsuleNetwork
from utils.image_processing import advanced_crack_refinement

IMG_SIZE = 448
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    """Loads the pre-trained model and sets it to evaluation mode."""
    model = EnhancedCrackCapsuleNetwork(img_size=IMG_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded successfully on", DEVICE)
    return model

def preprocess_image(image):
    """Prepares an image for the model, matching the validation transform."""
    transform = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transform(image=image)['image']

def postprocess_output(pred_logits, pred_width, original_size):
    """Converts model output tensors to final numpy masks."""
    pred_probs = torch.sigmoid(pred_logits)
    pred_mask_raw = (pred_probs > 0.5).cpu().numpy().squeeze().astype(np.uint8)
    
    pred_mask_refined = advanced_crack_refinement(pred_mask_raw)

    h, w = original_size
    output_mask = cv2.resize(pred_mask_refined, (w, h), interpolation=cv2.INTER_NEAREST)
    output_width_map = cv2.resize(pred_width.cpu().numpy().squeeze(), (w, h), interpolation=cv2.INTER_LINEAR)
    
    return output_mask, output_width_map

def quantify_cracks(mask, pred_width_map, px_to_mm_scale=0.1):
    """
    Quantifies crack properties from the final mask.
    NOTE: px_to_mm_scale is an assumption. For real-world use, this
    would need to be calibrated.
    """
    DENORMALIZATION_FACTOR = 40.0 
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8, cv2.CV_32S)
    
    crack_details = []
    if num_labels <= 1:
        return crack_details, None

    output_viz = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    for i in range(1, num_labels):
        component_mask = (labels == i)
        
        x, y, w, h, area = stats[i]
        
        contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        length_px = sum(cv2.arcLength(c, False) for c in contours) 
        length_mm = length_px * px_to_mm_scale
        
        width_values_normalized = pred_width_map[component_mask]
        if width_values_normalized.size == 0: continue
        
        width_values_denormalized = width_values_normalized * DENORMALIZATION_FACTOR
        width_values_mm = width_values_denormalized * px_to_mm_scale
        
        details = {
            'id': i,
            'length_mm': f"{length_mm:.2f}",
            'avg_width_mm': f"{np.mean(width_values_mm):.3f}",
            'max_width_mm': f"{np.max(width_values_mm):.3f}",
            'area_pixels': int(area)
        }
        crack_details.append(details)

        cv2.rectangle(output_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output_viz, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return crack_details, output_viz


def run_prediction(model, image_path):
    """Main prediction pipeline."""
    img_pil = Image.open(image_path).convert("RGB")
    original_size = img_pil.size[::-1]
    img_np = np.array(img_pil)

    img_tensor = preprocess_image(img_np).to(DEVICE).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        pred_logits = outputs['segmentation_logits']
        pred_width = outputs['width_map']

    final_mask, final_width_map = postprocess_output(pred_logits, pred_width, original_size)

    crack_data, quantified_viz = quantify_cracks(final_mask, final_width_map)
    
    overlay = img_np.copy()
    overlay[final_mask == 1, 1] = 255 
    
    return {
        'mask': final_mask,
        'overlay': overlay,
        'quantified_viz': quantified_viz,
        'analysis': crack_data
    }