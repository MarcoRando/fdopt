import torch
import numpy as np
import torch.nn as nn

from math import sqrt


class ThreeLayerNetwork(nn.Module):
    
    def __init__(self, in_features, 
                 out_features, hidden_sizes=[32, 32],
                 seed = 121212, dtype=torch.float32, 
                 device = 'cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(self.in_features, hidden_sizes[0], device=self.device, dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1], device=self.device, dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], out_features, device=self.device, dtype=self.dtype)
        ) 
        
    def forward(self, x):
        return self.model(x)
        
    def predict(self, X, scale='softmax'):
        X_norm = (X - X.mean())/(X.std() + 1e-5)
        raw = self.forward(X_norm)
        if scale == 'tanh':
            return nn.functional.tanh(raw)
        elif scale == 'softmax':
            probs = nn.functional.softmax(raw, dim=1, dtype=self.dtype)
            return probs.multinomial(1, generator=self.generator)
        return raw
        
        
        