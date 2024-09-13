import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import sparse
# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Function to create a sparse matrix
def create_sparse_matrix(rows, cols, density=0.5):
    matrix = sparse.random(rows, cols, density=density, format='csr', dtype=np.float32)
    return matrix

# Convert sparse matrix to torch tensor
def sparse_to_torch(sparse_matrix):
    coo = sparse_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape)).to_dense()

# Define contrastive loss function
def contrastive_loss(x1, x2, y, margin=1.0):
    dist = F.pairwise_distance(x1.unsqueeze(1), x2)
    loss = y * torch.pow(dist, 2) + (1 - y) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    return loss.mean()


def create_similar_matrix(A, perturbation_scale=0.1):
    B = A.copy()
    rows, cols = B.shape

    # Add small perturbations to non-zero elements
    for i in range(rows):
        for j in range(cols):
            if B[i, j] != 0:
                B[i, j] += np.random.normal(0, perturbation_scale * abs(B[i, j]))

    return B

# Function to calculate loss for a given e value
def calculate_loss(e,f, density=0.5, perturbation_scale=0.1):
    # Generate sparse matrices A and B (2x10)
    A = create_sparse_matrix(2, 1000, density)
    B = create_similar_matrix(A, perturbation_scale)

    # Generate sparse matrix C (8x10)
    C = create_sparse_matrix(100, 1000, density)

    # Set column 5 (index 4) to constant e for all matrices
    A[:, 4] = e
    B[:, 3] = f
    C[:, 3] = f

    # Convert to PyTorch tensors
    A_torch = sparse_to_torch(A)
    B_torch = sparse_to_torch(B)
    C_torch = sparse_to_torch(C)

    # Concatenate B and C
    BC_torch = torch.cat((B_torch, C_torch), dim=0)

    # Calculate contrastive loss
    y = torch.cat([torch.ones(2), torch.zeros(100)])
    loss = contrastive_loss(A_torch, BC_torch, y)

    return loss.item()

# Generate data for the plot
e_values = np.linspace(0, 5, 50)
losses = [calculate_loss(e, f=1.0,density=0.5) for e in e_values]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(e_values, losses)
plt.title('Contrastive Loss vs. Constant e (Sparse Matrices)')
plt.xlabel('Constant e')
plt.ylabel('Contrastive Loss')
plt.grid(True)

# Save the plot
plt.savefig('./vis/curve_e.png')
plt.close()


f_values = np.linspace(0, 5, 50)
losses = [calculate_loss(1, f, density=0.5) for f in f_values]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(f_values, losses)
plt.title('Contrastive Loss vs. Constant f (Sparse Matrices)')
plt.xlabel('Constant f')
plt.ylabel('Contrastive Loss')
plt.grid(True)

# Save the plot
plt.savefig('./vis/curve_f.png')
plt.close()

