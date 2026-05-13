import torch
import torch.nn as nn

class NewsGenerator(nn.Module):
    """
    The Forger: Creates synthetic 768-dimension news vectors.
    """
    def __init__(self, noise_dim=100, num_classes=3, output_dim=768):
        super(NewsGenerator, self).__init__()
        
        # We combine the random noise + the bias class (Neutral, Left, Right)
        self.input_dim = noise_dim + num_classes
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            # Output must match the IndicBERT embedding size (768)
            nn.Linear(512, output_dim),
            nn.Tanh() # Normalizes the fake vector between -1 and 1
        )
        self.device = self._get_device()
        self.to(self.device)

    def _get_device(self):
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    def forward(self, noise, labels):
        # Merge the noise and the requested label
        x = torch.cat([noise, labels], dim=1)
        return self.net(x)

class NewsDiscriminator(nn.Module):
    """
    The Detective: Tries to tell if a 768-dim vector is from a 
    real Indian news article or forged by the Generator.
    """
    def __init__(self, input_dim=768, num_classes=3):
        super(NewsDiscriminator, self).__init__()
        
        # Looks at the vector + the label it claims to be
        self.input_dim = input_dim + num_classes
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3), # Prevents the detective from memorizing
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Outputs a single number: 0 (Fake) to 1 (Real)
            nn.Linear(256, 1),
            nn.Sigmoid() 
        )
        self.device = self._get_device()
        self.to(self.device)

    def _get_device(self):
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    def forward(self, vectors, labels):
        x = torch.cat([vectors, labels], dim=1)
        return self.net(x)