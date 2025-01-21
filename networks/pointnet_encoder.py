import torch
import torch.nn as nn

class PointNetEncoder(nn.Module):
    def __init__(self, latent_size, input_channels=3):
        super(PointNetEncoder, self).__init__()
        
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        self.max_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_size)
        )

    def forward(self, x):
        x = x.float()
        x = x.transpose(2, 1)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        #print(f"Shape of x: {x.shape}")
        z = self.fc(x)
        return z