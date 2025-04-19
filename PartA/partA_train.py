import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, random_split
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.model_selection import StratifiedShuffleSplit

# Constants
IMG_DIM = 224
NUM_CLASSES = 10

def stratified_dataset_split(dataset, val_ratio=0.2):
    targets = [label for _, label in dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def create_data_pipelines(augment=False):
    base_transforms = [
        transforms.Resize((IMG_DIM, IMG_DIM)),
        transforms.ToTensor()
    ]
    
    if augment:
        augmentation_transforms = [
            transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9,1.1)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        ]
        base_transforms = augmentation_transforms + base_transforms
    
    return {
        'train': transforms.Compose(base_transforms),
        'test': transforms.Compose([
            transforms.Resize((IMG_DIM, IMG_DIM)),
            transforms.ToTensor()
        ])
    }

class CustomCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.features = self._create_conv_layers()
        self.classifier = self._create_fc_layers()
        
    def _create_conv_layers(self):
        layers = []
        in_channels = 3
        filters = [self.config['init_filters'] * (2**i) for i in range(self.config['conv_blocks'])]
        
        for idx, out_channels in enumerate(filters):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels) if self.config['use_bn'] else nn.Identity(),
                self._get_activation(self.config['activation']),
                nn.MaxPool2d(2),
                nn.Dropout2d(self.config['conv_dropout'])
            ]
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _create_fc_layers(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_DIM, IMG_DIM)
            features = self.features(dummy).view(1, -1)
            in_features = features.size(1)
        
        return nn.Sequential(
            nn.Linear(in_features, self.config['hidden_size']),
            nn.ReLU(),
            nn.Dropout(self.config['fc_dropout']),
            nn.Linear(self.config['hidden_size'], NUM_CLASSES)
        )
    
    def _get_activation(self, name):
        activations = {
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'mish': nn.Mish(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def model_evaluation(model, data_loader, device, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
    
    return {
        'loss': total_loss / len(data_loader),
        'accuracy': correct / len(data_loader.dataset)
    }

def training_cycle(config, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomCNN(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    best_val_acc = 0.0
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        epoch_loss, correct = 0.0, 0
        for inputs, labels in data['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
        
        train_metrics = {
            'loss': epoch_loss / len(data['train']),
            'accuracy': correct / len(data['train'].dataset)
        }
        
        # Validation phase
        val_metrics = model_evaluation(model, data['val'], device, criterion)
        scheduler.step(val_metrics['accuracy'])
        
        # Logging
        metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'lr': optimizer.param_groups[0]['lr']
        }
        wandb.log(metrics)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), 'best_model.pt')
    
    # Final evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    test_metrics = model_evaluation(model, data['test'], device, criterion)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    wandb.log({'test_acc': test_metrics['accuracy']})

def setup_data_loaders(config):
    transforms = create_data_pipelines(config['augment'])
    
    train_data = ImageFolder(config['train_dir'], transform=transforms['train'])
    train_set, val_set = stratified_dataset_split(train_data)
    test_set = ImageFolder(config['test_dir'], transform=transforms['test'])
    
    return {
        'train': DataLoader(train_set, batch_size=config['bs'], shuffle=True),
        'val': DataLoader(val_set, batch_size=config['bs']),
        'test': DataLoader(test_set, batch_size=config['bs'])
    }

def parse_args():
    parser = argparse.ArgumentParser(description='CNN Training Configuration')
    parser.add_argument('--project', type=str, default='BioCNN')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--init_filters', type=int, default=32)
    parser.add_argument('--conv_blocks', type=int, choices=[3,4,5], default=4)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--conv_dropout', type=float, default=0.3)
    parser.add_argument('--fc_dropout', type=float, default=0.5)
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--test_dir', required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    config = vars(args)
    
    wandb.init(project=args.project, config=config)
    data_loaders = setup_data_loaders(config)
    training_cycle(config, data_loaders)

if __name__ == '__main__':
    main()