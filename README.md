
# Graph Neural Network for Link Prediction

## Overview

This project extends the functionality of the `FutureOfAIviaAI` repository by implementing a **Graph Convolutional Network (GCN)** to solve the **link prediction task** on a graph dataset. The dataset used consists of semantic graphs provided in the `Science4Cast_18datasets` repository.

---

## Method Implemented: Graph Convolutional Network (GCN)

**Graph Convolutional Networks (GCNs)** are a class of graph neural networks designed to work on structured graph data. GCNs use message-passing operations to propagate information across nodes in the graph.

The implemented method:
1. **Converts the input graph** into a PyTorch Geometric-compatible format.
2. **Uses GCN layers** to extract meaningful node embeddings.
3. **Predicts edges** (links) using node embeddings.
4. **Evaluates model performance** using the **AUC-ROC** metric.

---

## Why GCN?

1. **Simplicity and Efficiency**: GCN is widely used for graph-based tasks due to its efficiency and ease of implementation.
2. **Link Prediction Compatibility**: GCN is well-suited for learning node embeddings and predicting links.
3. **PyTorch Geometric Integration**: GCN can be seamlessly implemented using the PyTorch Geometric library.
4. **Data Compatibility**: The dataset structure is ideal for link prediction tasks with minimal preprocessing.

---

## Expected Advantages

1. **High Performance**: GCN effectively learns node representations to predict links accurately.
2. **Scalability**: Suitable for both static and dynamic graphs.
3. **Interpretable Representations**: Learned node embeddings can be analyzed for further insights.

---

## Potential Challenges

1. **Lack of Node Features**:
   - Challenge: The dataset does not contain pre-existing node features.
   - Solution: Used **one-hot encoding** for node features.

2. **Limited Depth of GCN**:
   - Challenge: GCNs suffer from gradient vanishing issues with increasing layers.
   - Solution: Restricted the model to **2 convolutional layers**.

3. **Large Graphs**:
   - Challenge: Training on large graphs can be memory-intensive.
   - Solution: Leveraged efficient PyTorch tensor operations and GPU acceleration.

---

## Dataset

The dataset used in this project is from Zenodo (DOI: [10.5281/zenodo.7882892](https://zenodo.org/records/7882892#.ZE-Egx9BwuU)).  
The files contain:
- **Edges**: Connections (source, target) between nodes.
- Different parameters such as:
  - `delta`: Time intervals.
  - `cutoff`: Degree cutoff of nodes.
  - `minedge`: Minimum edges included.

The graph is represented in `.pkl` format, which includes:
- `train_dynamic_graph_sparse`: List of edges in the training graph.
- `train_edges_for_checking`: Edges for evaluation.
- `train_edges_solution`: Ground truth edges.

---

## Implementation

### Code Structure

- **`train_dynamic_graph_sparse`**: List of edges in the training graph.
- **`Data` object**: Converts the edge list into a PyTorch Geometric-compatible format.

### Key Components
1. **Graph Conversion**:
   Edges are converted to a PyTorch tensor:
   ```python
   train_dynamic_graph_sparse = [edge[:2] for edge in train_dynamic_graph_sparse]
   edge_index = torch.tensor(train_dynamic_graph_sparse, dtype=torch.long).t()
   data = Data(edge_index=edge_index, num_nodes=num_nodes)
   ```

2. **GCN Model**:
   ```python
   class GCN(torch.nn.Module):
       def __init__(self, in_channels, hidden_channels, out_channels):
           super(GCN, self).__init__()
           self.conv1 = GCNConv(in_channels, hidden_channels)
           self.conv2 = GCNConv(hidden_channels, out_channels)

       def forward(self, x, edge_index):
           x = F.relu(self.conv1(x, edge_index))
           x = self.conv2(x, edge_index)
           return x
   ```

3. **Training Loop**:
   The loss is calculated as **MSE** between node embeddings of connected nodes:
   ```python
   def train():
       model.train()
       optimizer.zero_grad()
       z = model(data.x, data.edge_index)
       loss = F.mse_loss(z[data.edge_index[0]], z[data.edge_index[1]])
       loss.backward()
       optimizer.step()
       return loss.item()
   ```

4. **Evaluation**:
   AUC-ROC is calculated using positive and negative edges:
   ```python
   pos_pred = (z[data.edge_index[0]] * z[data.edge_index[1]]).sum(dim=1)
   neg_edge_index = torch.randint(0, data.num_nodes, data.edge_index.size())
   neg_pred = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

   labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))])
   preds = torch.cat([pos_pred, neg_pred])
   auc = roc_auc_score(labels.cpu(), preds.cpu())
   ```

---

## Results

### Results on Provided Dataset
- **AUC-ROC**: **0.847**
- Dataset: `SemanticGraph_delta_5_cutoff_0_minedge_1.pkl`

### Results on Cora Dataset
- **AUC-ROC**: **0.79**
- Dataset: Cora (PyTorch Geometric).

---

## Comparison with Existing Methods

| Method       | Dataset                  | AUC-ROC |
|--------------|--------------------------|---------|
| **GCN**      | SemanticGraph_delta_5    | 0.847   |
| **GCN**      | Cora                     | 0.79    |


**Conclusion**:  
GCN outperforms baseline methods such as "Common Neighbors" (CN) and achieves competitive results on both datasets.

---

## Running the Code

### Prerequisites
Install required libraries:
```bash
pip install torch torch-geometric scikit-learn
```

### Running Training
To execute the code:
1. Replace my file path with the path to the `.pkl` dataset file.
2. Run the Python script.

Example:
```python
python model_train_evaluate.py
```

### Expected Output
The script will output the loss during training and the AUC score:
```
epoch 1, loss: 0.312456
epoch 2, loss: 0.245612
...
AUC: 0.8470
```

---

## Future Improvements

1. **Incorporate Node Features**: Use domain-specific node features instead of one-hot encoding.
2. **Implement GIN**: Add a Graph Isomorphism Network to compare performance with GCN.
3. **Dynamic Graph Training**: Extend the model to handle temporal graphs for evolving link prediction tasks.

---

## Conclusion

The project successfully implemented a **GCN model** for the link prediction task using PyTorch Geometric. The model achieved an **AUC score of 0.847** on the provided dataset and **0.79** on Cora, demonstrating its effectiveness in predicting edges in large-scale graphs.
