import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import global_mean_pool, knn_graph
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import MessagePassing


class GCN(torch.nn.Module):
    def __init__(self, numFeatures, numClasses):
        super(GCN, self).__init__()
        self.gn0 = GraphNorm(numFeatures)
        self.conv1 = GCNConv(numFeatures, 64)
        self.gn1 = GraphNorm(64)
        self.conv2 = GCNConv(64, 64)
        self.gn2 = GraphNorm(64)
        self.conv3 = GCNConv(64, 64)
        self.dense = Linear(64, 64)
        self.output = Linear(64, numClasses)

    def forward(self, x, edgeIndex, batch=None):
        # convolution layers
        x = self.gn0(x, batch)
        x = F.relu(self.conv1(x, edgeIndex))
        x = self.gn1(x, batch)
        x = F.relu(self.conv2(x, edgeIndex))
        x = self.gn2(x, batch)
        x = F.relu(self.conv3(x, edgeIndex))

        # readout layer
        x = global_mean_pool(x, batch)

        # dense layer
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.dense(x))
        x = self.output(x)

        return F.softmax(x, dim=1)


class GNN(torch.nn.Module):
    def __init__(self, numFeatures, numClasses):
        super(GNN, self).__init__()
        self.gn0 = GraphNorm(numFeatures)
        self.conv1 = GraphConv(numFeatures, 64)
        self.gn1 = GraphNorm(64)
        self.conv2 = GraphConv(64, 64)
        self.gn2 = GraphNorm(64)
        self.conv3 = GraphConv(64, 64)
        self.dense = Linear(64, 64)
        self.output = Linear(64, numClasses)

    def forward(self, x, edgeIndex, batch=None):
        # convolution layers
        x = self.gn0(x, batch)
        x = F.relu(self.conv1(x, edgeIndex))
        x = self.gn1(x, batch)
        x = F.relu(self.conv2(x, edgeIndex))
        x = self.gn2(x, batch)
        x = F.relu(self.conv3(x, edgeIndex))

        # readout layer
        x = global_mean_pool(x, batch)

        # dense layer
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.dense(x))
        x = self.output(x)

        return F.softmax(x, dim=1) 


class EdgeConv(MessagePassing):
    def __init__(self, inChannels, outChannels):
        super().__init__("mean")
        self.mlp = Sequential(
                Linear(2*inChannels, outChannels), BatchNorm1d(outChannels), ReLU(),
                Linear(outChannels, outChannels), BatchNorm1d(outChannels), ReLU(),
                Linear(outChannels, outChannels), BatchNorm1d(outChannels), ReLU())

    def forward(self, x, edgeIndex, batch=None):
        return self.propagate(edgeIndex, x=x, batch=batch)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_i-x_j], dim=1)
        return self.mlp(tmp)


class DynamicEdgeConv(EdgeConv):
    def __init__(self, inChannels, outChannels, k=4):
        super().__init__(inChannels, outChannels)
        self.shortcut = Sequential(Linear(inChannels, outChannels), BatchNorm1d(outChannels))
        self.k = k

    def forward(self, x, edgeIndex=None, batch=None):
        if not edgeIndex:
            edgeIndex = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        out = super().forward(x, edgeIndex, batch=batch)
        out += self.shortcut(x)
        return F.relu(out)


class ParticleNet(torch.nn.Module):
    def __init__(self, numFeatures, numClasses):
        super(ParticleNet, self).__init__()
        self.gn0 = GraphNorm(numFeatures)
        self.conv1 = DynamicEdgeConv(numFeatures, 64)
        self.gn1 = GraphNorm(64)
        self.conv2 = DynamicEdgeConv(64, 128)
        self.gn2 = GraphNorm(128)
        self.conv3 = DynamicEdgeConv(128, 128)
        self.gn3 = GraphNorm(128)
        self.dense1 = Linear(128, 64)
        self.dense2 = Linear(64, 64)
        self.output = Linear(64, numClasses)

    def forward(self, x, edgeIndex, batch=None):
        # convolution layers
        x = self.gn0(x, batch=batch)
        x = self.conv1(x, edgeIndex, batch=batch)
        x = self.gn1(x, batch=batch)
        x = self.conv2(x, batch=batch)
        x = self.gn2(x, batch=batch)
        x = self.conv3(x, batch=batch)
        x = self.gn3(x, batch=batch)

        # readout layer
        x = global_mean_pool(x, batch=batch)

        # dense layer
        x = F.relu(self.dense1(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.dense2(x))
        x = F.dropout(x, p=0.2)
        x = self.output(x)

        return F.softmax(x, dim=1)


class ParticleNetLite(torch.nn.Module):
    def __init__(self, numFeatures, numClasses):
        super(ParticleNet, self).__init__()
        self.gn0 = GraphNorm(numFeatures)
        self.conv1 = DynamicEdgeConv(numFeatures, 32)
        self.gn1 = GraphNorm(32)
        self.conv2 = DynamicEdgeConv(32, 64)
        self.gn2 = GraphNorm(64)
        self.conv3 = DynamicEdgeConv(64, 64)
        self.gn3 = GraphNorm(64)
        self.dense1 = Linear(64, 32)
        self.dense2 = Linear(32, 32)
        self.output = Linear(32, numClasses)

    def forward(self, x, edgeIndex, batch=None):
        # convolution layers
        x = self.gn0(x, batch=batch)
        x = self.conv1(x, edgeIndex, batch=batch)
        x = self.gn1(x, batch=batch)
        x = self.conv2(x, batch=batch)
        x = self.gn2(x, batch=batch)
        x = self.conv3(x, batch=batch)
        x = self.gn3(x, batch=batch)

        # readout layer
        x = global_mean_pool(x, batch=batch)

        # dense layer
        x = F.relu(self.dense1(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.dense2(x))
        x = F.dropout(x, p=0.2)
        x = self.output(x)

        return F.softmax(x, dim=1)
