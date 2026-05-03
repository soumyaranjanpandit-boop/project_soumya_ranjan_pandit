import os
import torch
from dataset import get_dataloader
from model import TumorClassifier
import torchvision.transforms as transforms
from PIL import Image
import config
import numpy as np
import cv2

def get_transform():
    return transforms.Compose([
        transforms.Resize((config.resize_x, config.resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_item(input_path='./data', weights_path='./checkpoints/final_weights.pth', *args, **kwargs):
    """
    Can take a directory path, a single image file path, or a list of image file paths.
    Returns inferences for the batch/item.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Predicting on device: {device}")
    
    # Instantiate Model
    model = TumorClassifier(num_classes=2)
    
    if not os.path.exists(weights_path):
        print(f"Weights file not found at {weights_path}")
        return None
        
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    
    # If it's a directory, use the dataloader approach
    if isinstance(input_path, str) and os.path.isdir(input_path):
        test_loader = get_dataloader(input_path, batch_size=8, shuffle=False)
        if test_loader is None:
            return None
            
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                for i in range(len(predicted)):
                    pred_class = predicted[i].item()
                    prob = probs[i][pred_class].item()
                    label_str = "Tumor" if pred_class == 1 else "No Tumor"
                    all_predictions.append((label_str, prob))
                    print(f"Prediction: {label_str} (Confidence: {prob*100:.2f}%)")
                    
        return all_predictions

    # If it's a single file or a list of files
    if isinstance(input_path, str) and os.path.isfile(input_path):
        input_paths = [input_path]
    elif isinstance(input_path, list):
        input_paths = input_path
    else:
        print("Invalid input_path format.")
        return None

    transform = get_transform()
    
    with torch.no_grad():
        for img_path in input_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension
                
                # Make sure input requires grad for Grad-CAM
                input_tensor.requires_grad_()
                
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                pred_class = predicted[0].item()
                prob = probs[0][pred_class].item()
                label_str = "Tumor" if pred_class == 1 else "No Tumor"
                all_predictions.append((img_path, label_str, prob))
                print(f"File: {os.path.basename(img_path)} | Prediction: {label_str} (Confidence: {prob*100:.2f}%)")
                
                # --- Grad-CAM Generation ---
                # Backpropagate to get gradients
                model.zero_grad()
                outputs[0, pred_class].backward()
                
                gradients = model.gradients.cpu().data.numpy()
                activations = model.activations.cpu().data.numpy()
                
                # Global average pooling of gradients
                weights = np.mean(gradients, axis=(2, 3))[0, :]
                
                # Weighted sum of activations
                cam = np.zeros(activations.shape[2:], dtype=np.float32)
                for i, w in enumerate(weights):
                    cam += w * activations[0, i, :, :]
                    
                cam = np.maximum(cam, 0) # ReLU
                if np.max(cam) > 0:
                    cam = cam / np.max(cam)
                    
                # Resize and overlay
                original_img = cv2.imread(img_path)
                original_img = cv2.resize(original_img, (config.resize_x, config.resize_y))
                cam_resized = cv2.resize(cam, (config.resize_x, config.resize_y))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                
                overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
                cam_save_path = f"gradcam_results_{os.path.basename(img_path)}.png"
                cv2.imwrite(cam_save_path, overlay)
                print(f"Saved Grad-CAM heatmap to {cam_save_path}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    return all_predictions

if __name__ == '__main__':
    predict_item(input_path='./data', weights_path='./checkpoints/final_weights.pth')
