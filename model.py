import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class TumorClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(TumorClassifier, self).__init__()
        # Load pretrained MobileNetV2
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Freeze the feature extraction layers
        for param in self.model.features.parameters():
            param.requires_grad = False
            
        # Modify the classifier head for our number of classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features, num_classes)
        )
        
        # Grad-CAM components
        self.gradients = None
        self.activations = None
        
        # Hook the last convolutional layer (features[-1] in MobileNetV2)
        target_layer = self.model.features[-1]
        for param in target_layer.parameters():
            param.requires_grad = True
            
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        return self.model(x)
