import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch_geometric.data import Data, Batch
import numpy as np
from skimage import segmentation
import redis
from torch_geometric.utils import to_dense_batch
import math
from tqdm import tqdm
# import cupy as cp
from fast_slic.avx2 import SlicAvx2

debug = False

def closest_factors(n):
    """Find the two closest factors of n that form a rectangle."""
    sqrt_n = int(math.sqrt(n))
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return i, n // i
    return 1, n

def image_to_graph(images, n_segments):
    graphs = []
    batch_size = images.size(0)

    for i in tqdm(range(batch_size), desc='image-to-graph'):
        # Extract image in (C, H, W) format and convert to NumPy array
        image = images[i].detach().cpu().numpy()  # CHW format

        # Reorder to (H, W, C) for segmentation and ensure C-contiguity
        image_hw_c = np.ascontiguousarray(image.transpose(1, 2, 0) * 255).astype(np.uint8)

        # Perform SLIC segmentation using fast_slic
        slic = SlicAvx2(num_components=n_segments, compactness=30)
        segments = slic.iterate(image_hw_c)
        num_segments = segments.max() + 1

        # Initialize node features
        nodes = np.zeros((num_segments, image.shape[0]), dtype=np.float32)
        edges_set = set()  # Use a set to avoid duplicate edges

        for j in range(num_segments):
            mask = segments == j

            if np.any(mask):  # Check if the mask is not empty
                # Compute mean only over the mask using efficient array indexing
                nodes[j] = np.mean(image[:, mask], axis=1) / 255.0  # Normalize to [0, 1]
            else:
                nodes[j] = np.zeros(image.shape[0], dtype=np.float32)

            # Identify neighbors more efficiently
            boundary_mask = segmentation.find_boundaries(mask, connectivity=1, mode='inner')
            boundary_neighbors = segments[boundary_mask]

            for neighbor in np.unique(boundary_neighbors):
                if neighbor != j:
                    edges_set.add((j, neighbor))
                    edges_set.add((neighbor, j))

        # Convert set to a list of edges
        if len(edges_set) > 0:
            edges = list(edges_set)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # Handle the case where no edges are found
            edge_index = torch.empty((2, 0), dtype=torch.long)  # Empty edge index
            
        x = torch.tensor(nodes, dtype=torch.float)

        if debug:
            print(f"Feature shape (data.x): {x.shape}")  # Should be (n_segments, C)
            print(f"Edge index shape (data.edge_index): {edge_index.shape}")  # Should be (2, number_of_edges)

        graph = Data(x=x, edge_index=edge_index)
        graphs.append(graph)

    batched_graph = Batch.from_data_list(graphs)
    return batched_graph


# Define GNNFeatureExtraction
class GNNFeatureExtraction(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, use_cuda):
        super(GNNFeatureExtraction, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
        self.conv3 = GCNConv(hidden_channels[1], out_channels)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.to('cuda')  # Move the entire model to GPU

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Ensure tensors are on the same device
        if self.use_cuda:
            x = x.to('cuda')
            edge_index = edge_index.to('cuda')

        if debug:
            print(f"x shape before conv1: {x.shape}")  # Debugging: print shape of x
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

# Define FeatureL2Norm
class FeatureL2Norm(nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.sqrt(torch.sum(feature ** 2, dim=1, keepdim=True) + epsilon)
        feature = feature / norm
        return feature


# Define Feature Correlation
class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B, batch_A, batch_B):
        feature_A, mask_A = to_dense_batch(feature_A, batch_A)
        feature_B, mask_B = to_dense_batch(feature_B, batch_B)

        if debug:
            print(f"Feature A shape: {feature_A.shape}")
            print(f"Feature B shape: {feature_B.shape}")

        # Find the maximum number of nodes
        max_nodes = max(feature_A.size(1), feature_B.size(1))

        # Pad feature_A and feature_B to have the same number of nodes
        if feature_A.size(1) < max_nodes:
            padding_size = max_nodes - feature_A.size(1)
            padding = torch.zeros((feature_A.size(0), padding_size, feature_A.size(2)), device=feature_A.device)
            feature_A = torch.cat([feature_A, padding], dim=1)
            mask_A = torch.cat([mask_A, torch.zeros((mask_A.size(0), padding_size), device=mask_A.device)], dim=1)

        if feature_B.size(1) < max_nodes:
            padding_size = max_nodes - feature_B.size(1)
            padding = torch.zeros((feature_B.size(0), padding_size, feature_B.size(2)), device=feature_B.device)
            feature_B = torch.cat([feature_B, padding], dim=1)
            mask_B = torch.cat([mask_B, torch.zeros((mask_B.size(0), padding_size), device=mask_B.device)], dim=1)

        # Perform batch matrix multiplication to get the correlation matrix
        correlation_tensor = torch.bmm(feature_A.transpose(1, 2), feature_B)

        if debug:
            print(f"Correlation tensor shape: {correlation_tensor.shape}")

        return correlation_tensor


# Define Regression Layer for Affine and TPS
class RegressionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_cuda, num_nodes):
        super(RegressionLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=4, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.linear = nn.Linear(64, output_dim)
        self.use_cuda = use_cuda
        self.output_dim = output_dim

        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, x):
        # Calculate the height and width
        batch_size = x.size(0)
        num_nodes = x.size(1)
        num_features = x.size(2)

        if debug:
            print(f"Original x shape: {x.shape}")

        # Get the closest factors for the height and width
        height, width = closest_factors(num_nodes)
        padding = height * width - num_nodes

        # Pad x to fit the grid dimensions
        if padding > 0:
            x = F.pad(x, (0, 0, 0, padding))

        # Reshape x to (batch_size, num_features, height, width)
        x = x.view(batch_size, num_features, height, width)

        if debug:
            print(f"Reshaped x shape for conv: {x.shape}")

        x = self.conv(x)
        x = x.view(x.size(0), -1)

        if debug:
            print(f"Flattened x shape for linear: {x.shape}, {tuple(x.size())}")

        x = self.linear(x)
        return x

# Define ProsGraphNet with GNN, FeatureL2Norm, Feature Correlation, and Regression
class ProsGraphNet(nn.Module):
    def __init__(self, geometric_model='affine', normalize_features=True, normalize_matches=True, use_cuda=True, num_of_segments=80):
        super(ProsGraphNet, self).__init__()
        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.geometric_model = geometric_model
        
        self.feature_extraction = GNNFeatureExtraction(in_channels=3, hidden_channels=[32,24], out_channels=16, use_cuda=use_cuda)
        self.l2_norm = FeatureL2Norm()
        self.feature_correlation = FeatureCorrelation()
        
        if geometric_model == 'affine':
            output_dim = 6
        elif geometric_model == 'tps':
            output_dim = 72
        
        self.feature_regression = RegressionLayer(input_dim=16, output_dim=output_dim, use_cuda=use_cuda, num_nodes=num_of_segments)
        self.ReLU = nn.ReLU(inplace=True)
        self.num_of_segments = num_of_segments

    def forward(self, tnf_batch):
        source_image = tnf_batch['source_image']
        target_image = tnf_batch['target_image']
        
        # Convert images to graphs
        source_graph = image_to_graph(source_image, self.num_of_segments)
        target_graph = image_to_graph(target_image, self.num_of_segments)
        
        # Move graphs to GPU if available
        if self.use_cuda:
            source_graph = source_graph.to('cuda')
            target_graph = target_graph.to('cuda')
        
        if debug:
            print(f"Graph Batched size {source_graph.size}, {source_graph.size}")
        # Extract features using GNN
        feature_A = self.feature_extraction(source_graph)
        feature_B = self.feature_extraction(target_graph)

        if debug:
            print(f"Feature Extracted Batched size {feature_A.shape}, {feature_B.shape}")
        
        # Normalize features
        if self.normalize_features:
            feature_A = self.l2_norm(feature_A)
            feature_B = self.l2_norm(feature_B)

        if debug:
            print(f"L2 norm Batched size {feature_A.size()}, {feature_B.size()}")
            print(f"Source Graph Batch: {source_graph.batch.size()}, Target Graph Batch: {target_graph.batch.size()}")
        
        # Compute feature correlation
        correlation = self.feature_correlation(feature_A, feature_B, source_graph.batch, target_graph.batch)

        if debug:
            print(f"Correlation Before Relu and L2 norm size {correlation.size()}")
        
        # Normalize matches
        if self.normalize_matches:
            correlation = self.l2_norm(self.ReLU(correlation))
        
        if debug:
            print(f"Norm Batched size {feature_A.size()}, {feature_B.size()}")

        # Perform regression to tnf parameters theta
        theta = self.feature_regression(correlation)
        
        # Adjust theta for affine and TPS
        if theta.shape[1] == 6:
            temp = torch.tensor([1.0, 0, 0, 0, 1.0, 0], device=theta.device)
            adjust = temp.repeat(theta.shape[0], 1)
            theta = 0.1 * theta + adjust
            theta = theta.view(theta.size()[0], 2, 3)
        
        elif theta.shape[1] == 72:
            temp = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                                 -0.6, -0.6, -0.6, -0.6, -0.6, -0.6,
                                 -0.2, -0.2, -0.2, -0.2, -0.2, -0.2,
                                  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,
                                  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,
                                  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                                 -1.0, -0.6, -0.2,  0.2,  0.6,  1.0,
                                 -1.0, -0.6, -0.2,  0.2,  0.6,  1.0,
                                 -1.0, -0.6, -0.2,  0.2,  0.6,  1.0,
                                 -1.0, -0.6, -0.2,  0.2,  0.6,  1.0,
                                 -1.0, -0.6, -0.2,  0.2,  0.6,  1.0,
                                 -1.0, -0.6, -0.2,  0.2,  0.6,  1.0], device=theta.device)
            adjust = temp.repeat(theta.shape[0], 1)
            theta = 0.1 * theta + adjust
        
        return theta

if __name__ == "__main__":
    source_image = np.random.rand(256, 256, 3)  # Replace with actual MRI image
    target_image = np.random.rand(256, 256, 3)  # Replace with actual MRI image

    tnf_batch = {'source_image': torch.tensor(source_image, dtype=torch.float32), 'target_image': torch.tensor(target_image, dtype=torch.float32)}

    model = ProsGraphNet(geometric_model='affine', use_cuda=torch.cuda.is_available())
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    theta = model(tnf_batch)
    print(theta)
