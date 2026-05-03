import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import config

class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filepaths = []
        self.labels = []
        self.transform = transform
        
        # We assume the directory has subdirectories for classes, just like the raw dataset
        valid_classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
        classes = os.listdir(data_dir)
        for cls in classes:
            if cls.lower() not in valid_classes:
                continue
                
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
                
            # Determine strict binary label
            if cls.lower() == 'no_tumor':
                label = 0 # No Tumor
            else:
                label = 1 # Tumor (glioma, meningioma, pituitary)
                
            for img_path in glob.glob(os.path.join(cls_dir, '*.*')):
                if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.filepaths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloader(data_dir='./data', batch_size=config.batch_size, shuffle=True, *args, **kwargs):
    transform = transforms.Compose([
        transforms.Resize((config.resize_x, config.resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = BrainTumorDataset(data_dir, transform=transform)
    if len(dataset) == 0:
        print(f"Warning: No images found in {data_dir}")
        return None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
