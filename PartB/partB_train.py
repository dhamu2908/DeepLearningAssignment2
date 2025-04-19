import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
import argparse

# Configuration handler
def setup_arguments():
    parser = argparse.ArgumentParser(description="Image Classification Trainer")
    parser.add_argument("-e", "--epochs", type=int, default=5, 
                       help="Training epochs count")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                       help="Size of mini-batches for training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                       help="Initial learning rate")
    parser.add_argument("-train", "--train_folder", required=True,
                       help="Path to training data directory")
    parser.add_argument("-test", "--test_folder", required=True,
                       help="Path to evaluation data directory")
    return parser.parse_args()

# Data preparation
def prepare_datasets(train_dir, test_dir):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_train = datasets.ImageFolder(train_dir, transform=preprocess)
    test_data = datasets.ImageFolder(test_dir, transform=preprocess)
    
    return full_train, test_data

# Model initialization
def initialize_model(class_count):
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in net.parameters():
        param.requires_grad = False  # Freeze base layers
        
    final_layer = nn.Linear(net.fc.in_features, class_count)
    net.fc = final_layer  # Replace classifier
    return net

# Training process
def execute_training(model, device, loss_fn, optimizer, 
                    train_loader, valid_loader, test_loader, epochs):
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-"*20)
        
        # Training phase
        model.train()
        current_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            current_loss += loss.item()
            if batch_idx % 50 == 49:
                print(f"Batch {batch_idx+1} | Loss: {current_loss/50:.4f}")
                current_loss = 0.0
        
        # Validation phase
        model.eval()
        val_loss, val_acc = evaluate_model(model, device, loss_fn, valid_loader)
        print(f"Validation Results - Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
    
    # Final evaluation
    print("\nTesting Model...")
    test_loss, test_acc = evaluate_model(model, device, loss_fn, test_loader)
    print(f"Test Results - Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

# Evaluation helper
def evaluate_model(model, device, loss_fn, data_loader):
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            running_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            correct_preds += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = running_loss / len(data_loader)
    accuracy = (correct_preds / total_samples) * 100
    return avg_loss, accuracy

if __name__ == "__main__":
    # Configuration setup
    config = setup_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data preparation
    train_data, test_set = prepare_datasets(config.train_folder, config.test_folder)
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_set, val_set = random_split(train_data, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(val_set, batch_size=config.batch_size)
    test_loader = DataLoader(test_set, batch_size=config.batch_size)
    
    # Model configuration
    network = initialize_model(len(train_data.dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.fc.parameters(), lr=config.learning_rate)
    
    # Start training process
    execute_training(
        network, device, criterion, optimizer,
        train_loader, valid_loader, test_loader,
        config.epochs
    )