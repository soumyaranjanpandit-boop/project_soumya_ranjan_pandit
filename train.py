import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from model import TumorClassifier
import config

def train_model(data_dir='./data', save_path='./checkpoints/final_weights.pth', *args, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Instantiate Model
    model = TumorClassifier(num_classes=2)
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Dataloader
    train_loader = get_dataloader(data_dir, batch_size=config.batch_size, shuffle=True)
    if train_loader is None:
        print("Dataloader is empty, training cannot proceed.")
        return
        
    print(f"Starting training for {config.number_of_epochs} epochs...")
    
    for epoch in range(config.number_of_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{config.number_of_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
        
    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Weights saved to {save_path}")

if __name__ == '__main__':
    train_model(data_dir='./data')
