
import torch
import torch.nn as nn

class WordScoringModel(nn.Module):
    def __init__(self, input_dim=184, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        '''
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        '''
    def forward(self, x):
        return self.net(x).squeeze(-1)
    
def get_hidden_activations(model, x_input):
    activations = []
    x = x_input

    # Iterate through model layers manually
    for layer in model.net:
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            activations.append(x.detach().clone())  # Save after ReLU

    return activations 


class ValueNet(nn.Module):
    def __init__(self, state_dim=28, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, s):
        return self.net(s).squeeze(-1)
