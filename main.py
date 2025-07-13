import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch_geometric.data import Data, Batch
import numpy as np
from skimage import segmentation
import redis


def image_to_graph(images, n_segments=400):
    graphs = []
    batch_size = images.size(0)

    for i in range(batch_size):
        image = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HxWxC
        segments = segmentation.slic(image, compactness=30, n_segments=n_segments)
        num_segments = segments.max() + 1

        nodes = np.zeros((num_segments, image.shape[2]), dtype=np.float32)
        edges = []

        for j in range(num_segments):
            mask = segments == j
            nodes[j] = image[mask].mean(axis=0)

            for neighbor in range(j + 1, num_segments):
                if np.any(segmentation.find_boundaries(segments == j) & (segments == neighbor)):
                    edges.append([j, neighbor])
                    edges.append([neighbor, j])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(nodes, dtype=torch.float)

        print(f"Feature shape (data.x): {x.shape}")  # Should be (n_segments, image.shape[2])
        print(f"Edge index shape (data.edge_index): {edge_index.shape}")  # Should be (2, number_of_edges)

        graph = Data(x=x, edge_index=edge_index)
        graphs.append(graph)

    batched_graph = Batch.from_data_list(graphs)
    return batched_graph


def main():
    print("waiting for batch...")
    rc = redis.StrictRedis("ecs-redis-cache.ecs-redis-cluster", port=6379, db=0, health_check_interval=30)
    processed_batches = []
    while 