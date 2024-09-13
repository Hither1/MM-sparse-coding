import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np


# Function to generate a matrix with one column being a constant
def generate_matrix(rows, cols, constant_value, constant_col_index):
    # Generate a random matrix
    matrix = np.random.randn(rows, cols)
    
    # Set the specified column to the constant value
    matrix[:, constant_col_index] = constant_value
    
    return matrix

# Parameters
rows = 5
cols = 4
constant_value = 3.14
constant_col_index = 2

# Generate the matrix
matrix = generate_matrix(rows, cols, constant_value, constant_col_index)
print("Generated Matrix:")
print(matrix)


# Simple Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Sample Data
data = np.random.randn(100, 10)
labels = np.random.randint(0, 2, size=(100,))

dataset = SimpleDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model, Loss, Optimizer
model = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


