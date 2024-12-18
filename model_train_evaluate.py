import pickle
import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score

file_path = "/SemanticGraph_delta_5_cutoff_0_minedge_1.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

train_dynamic_graph_sparse, train_edges_for_checking, train_edges_solution, year_start, current_delta, curr_vertex_degree_cutoff, current_min_edges = data

print("Amount of edges:", len(train_dynamic_graph_sparse))
print("First 5 edges:", train_dynamic_graph_sparse[:5])
print("Amount of edges for checking:", len(train_edges_for_checking))
print("Params", year_start, current_delta, curr_vertex_degree_cutoff, current_min_edges)



#Convert dataset to torch format
train_dynamic_graph_sparse = [edge[:2] for edge in train_dynamic_graph_sparse] 
edge_index = torch.tensor(train_dynamic_graph_sparse, dtype=torch.long).t()

num_nodes = edge_index.max().item() + 1

data = Data(edge_index=edge_index, num_nodes=num_nodes)

print("Graph", data)

data.x = torch.eye(data.num_nodes)  # One-hot features




# model GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

model = GCN(in_channels=data.num_nodes, hidden_channels=64, out_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)



def train():
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)
    loss = F.mse_loss(z[data.edge_index[0]], z[data.edge_index[1]])
    loss.backward()
    optimizer.step()
    return loss.item()


for epoch in range(1, 10):  # 50 эпох
    loss = train()
    print(f"epoch {epoch}, loss: {loss:.6f}")




model.eval()
with torch.no_grad():
    z = model(data.x, data.edge_index)
    pos_pred = (z[data.edge_index[0]] * z[data.edge_index[1]]).sum(dim=1)
    neg_edge_index = torch.randint(0, data.num_nodes, data.edge_index.size())  
    neg_pred = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)


labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))])
preds = torch.cat([pos_pred, neg_pred])

auc = roc_auc_score(labels.cpu(), preds.cpu())
print(f"AUC: {auc:.4f}")

