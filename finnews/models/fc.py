import torch.nn as nn
from torch.utils.data import Dataset


class FCBinaryClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim=1,
        dropout_rate=0.1
    ):
        """
        Fully Connected Neural Network for Binary Classification.

        Args:
            input_dim: Number of input features
            hidden_dims: List of integers specifying the hidden layer dimensions
            output_dim: Output dimension (defaults to 1 for binary classification)
        """
        super(FCBinaryClassifier, self).__init__()
        layers = []
        # Input to first hidden layer
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())  # Activation
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        # Last layer to output
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Sigmoid())  # Binary classification output
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FCBinaryDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
