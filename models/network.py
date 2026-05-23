from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet1D(nn.Module):
    def __init__(self, in_channels: int = 19, num_classes: int = 2, base: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, base, kernel_size=7, padding=3),
            nn.BatchNorm1d(base), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(base, base*2, kernel_size=7, padding=3),
            nn.BatchNorm1d(base*2), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(base*2, base*4, kernel_size=5, padding=2),
            nn.BatchNorm1d(base*4), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(base*4, base*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base*4), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base*4, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T)
        z = self.features(x)
        return self.head(z)


class InceptionModule1D(nn.Module):
    """Inception module for 1D EEG signals"""
    def __init__(self, in_channels, out_channels_list):
        super().__init__()
        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_list[0], kernel_size=1),
            nn.BatchNorm1d(out_channels_list[0]), nn.ReLU()
        )
        
        # Branch 2: 1x1 conv + 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_list[1], kernel_size=1),
            nn.BatchNorm1d(out_channels_list[1]), nn.ReLU(),
            nn.Conv1d(out_channels_list[1], out_channels_list[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels_list[2]), nn.ReLU()
        )
        
        # Branch 3: 1x1 conv + 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_list[3], kernel_size=1),
            nn.BatchNorm1d(out_channels_list[3]), nn.ReLU(),
            nn.Conv1d(out_channels_list[3], out_channels_list[4], kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels_list[4]), nn.ReLU()
        )
        
        # Branch 4: 3x3 max pool + 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels_list[5], kernel_size=1),
            nn.BatchNorm1d(out_channels_list[5]), nn.ReLU()
        )
    
    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)


class EEGGoogLeNet(nn.Module):
    """GoogLeNet-inspired architecture for 1D EEG signals"""
    def __init__(self, in_channels: int = 19, num_classes: int = 2):
        super().__init__()
        
        # Initial conv layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception modules
        self.inception1 = InceptionModule1D(64, [16, 16, 32, 16, 32, 16])
        self.inception2 = InceptionModule1D(96, [24, 24, 48, 24, 48, 24])
        self.inception3 = InceptionModule1D(144, [32, 32, 64, 32, 64, 32])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(192, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        return self.classifier(x)


class DenseBlock1D(nn.Module):
    """Dense block for 1D signals"""
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm1d(current_channels),
                nn.ReLU(),
                nn.Conv1d(current_channels, growth_rate, kernel_size=3, padding=1)
            ))
            current_channels += growth_rate
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


class EEGDenseNet(nn.Module):
    """Simplified DenseNet-inspired architecture for 1D EEG signals"""
    def __init__(self, in_channels: int = 19, num_classes: int = 2):
        super().__init__()
        
        # Initial conv
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense-like blocks (simplified to avoid concatenation issues)
        self.block1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU()
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


class EEGVGG(nn.Module):
    """VGG-inspired architecture for 1D EEG signals"""
    def __init__(self, in_channels: int = 19, num_classes: int = 2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ResidualBlock1D(nn.Module):
    """Residual block for 1D signals"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return F.relu(x)


class EEGResNet(nn.Module):
    """ResNet-inspired architecture for 1D EEG signals"""
    def __init__(self, in_channels: int = 19, num_classes: int = 2):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual layers
        self.layer1 = nn.Sequential(
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock1D(128, 256, stride=2),
            ResidualBlock1D(256, 256)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)


class EEGRNN(nn.Module):
    """RNN-based architecture for EEG signals"""
    def __init__(self, in_channels: int = 19, hidden_size: int = 128, num_classes: int = 2):
        super().__init__()
        
        # Feature extraction
        self.conv_features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # RNN layers
        self.rnn = nn.LSTM(64, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        # x: (N, C, T)
        x = self.conv_features(x)  # (N, 64, T')
        x = x.transpose(1, 2)  # (N, T', 64)
        
        # RNN
        rnn_out, (hidden, cell) = self.rnn(x)
        # Use last output
        last_output = rnn_out[:, -1, :]
        
        return self.classifier(last_output)


class DeepCNN1D(nn.Module):
    """Deep 1D CNN with modern architecture"""
    def __init__(self, in_channels: int = 19, num_classes: int = 2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.2),
            
            # Block 2
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.3),
            
            # Block 3
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.4),
            
            # Block 4
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# Model factory function
def create_model(model_name: str, in_channels: int = 19, num_classes: int = 2, **kwargs):
    """Create a model by name"""
    models = {
        'eegnet': EEGNet1D,
        'googlenet': EEGGoogLeNet,
        'densenet': EEGDenseNet,
        'vgg': EEGVGG,
        'resnet': EEGResNet,
        'rnn': EEGRNN,
        'deepcnn': DeepCNN1D
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](in_channels=in_channels, num_classes=num_classes, **kwargs)
