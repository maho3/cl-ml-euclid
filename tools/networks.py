import torch
from torch_geometric import nn as gnn
from torch_geometric.nn import aggr
import torch.nn.functional as F


class DenseNetwork(torch.nn.Module):
    def __init__(self, in_channels, dense_channels, out_channels, drop_p=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.dense_channels = dense_channels
        self.out_channels = out_channels
        self.drop_p = drop_p

        self.global_aggr = aggr.MultiAggregation(
            aggrs=['sum', 'mean', 'std', 'min', 'max'],
            mode='cat'
        )
        self._build_dnn(in_channels*len(self.global_aggr.aggrs))
        self.dropout = torch.nn.Dropout(drop_p)

    def _build_dnn(self, in_channels):
        self.fc1 = torch.nn.Linear(in_channels, self.dense_channels[0])
        self.fcs = torch.nn.ModuleList(
            [torch.nn.Linear(self.dense_channels[i], self.dense_channels[i+1])
             for i in range(0, len(self.dense_channels)-1)])
        self.fc2 = torch.nn.Linear(self.dense_channels[-1], self.out_channels)

    def dnn(self, x):
        x = F.relu(self.fc1(x))
        for fc in self.fcs:
            x = self.dropout(x)
            x = F.relu(fc(x))
        x = self.fc2(x)
        return x

    def global_features(self, x, ptr=None):
        x = self.global_aggr(x, ptr=ptr)
        x = torch.nan_to_num(x, nan=1., posinf=1., neginf=1.)
        return x

    def forward(self, x, edge_index):
        x = self.global_features(x)
        x = self.dnn(x)
        return x


class GATNetwork(DenseNetwork):
    def __init__(
        self, in_channels, gcn_channels, gcn_heads,
        dense_channels, out_channels, drop_p=0.1
    ):
        super().__init__(in_channels, dense_channels, out_channels, drop_p)
        self.gcn_channels = gcn_channels
        self.gcn_heads = gcn_heads

        self.graph_aggr = aggr.MultiAggregation(
            aggrs=['sum', 'mean', 'std', 'min',
                   'max', aggr.SoftmaxAggregation(learn=True)],
            mode='cat'
        )

        self._build_gnn()
        self._build_dnn(gcn_channels[-1]*len(self.graph_aggr.aggrs) +
                        in_channels*len(self.global_aggr.aggrs))

    def _build_gnn(self):
        self.conv1 = gnn.GATv2Conv(
            self.in_channels, self.gcn_channels[0], heads=self.gcn_heads[0])
        self.convs = torch.nn.ModuleList(
            [gnn.GATv2Conv(self.gcn_channels[i]*self.gcn_heads[i],
                           self.gcn_channels[i+1], heads=self.gcn_heads[i+1])
             for i in range(len(self.gcn_channels)-2)]
        )
        self.conv2 = gnn.GATv2Conv(
            self.gcn_channels[-2]*self.gcn_heads[-2],
            self.gcn_channels[-1], heads=self.gcn_heads[-1], concat=False)

    def gnn(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    def forward(self, data):
        features = data.x.float()
        edge_index = data.edge_index.int()
        if hasattr(data, 'ptr'):
            ptr = data.ptr
        else:
            ptr = None

        globfeat = self.global_features(features, ptr)

        x = self.gnn(features, edge_index)
        x = self.graph_aggr(x, ptr=ptr)
        x = torch.cat([x, globfeat], dim=-1)
        x = self.dnn(x)
        return x


class SAGENetwork(DenseNetwork):
    def __init__(
        self, in_channels, gcn_channels,
        dense_channels, out_channels, drop_p=0.1
    ):
        super().__init__(in_channels, dense_channels, out_channels, drop_p)
        self.gcn_channels = gcn_channels

        self.graph_aggr = aggr.MultiAggregation(
            aggrs=['sum', 'mean', 'median', 'std', 'min',
                   'max', aggr.SoftmaxAggregation(learn=True)],
            mode='cat'
        )

        self._build_gnn()
        self._build_dnn(gcn_channels[-1]*len(self.graph_aggr.aggrs) +
                        in_channels*len(self.global_aggr.aggrs))

    def _build_gnn(self):
        self.conv1 = gnn.SAGEConv(self.in_channels, self.gcn_channels[0])
        self.convs = torch.nn.ModuleList(
            [gnn.SAGEConv(self.gcn_channels[i], self.gcn_channels[i+1])
             for i in range(len(self.gcn_channels)-1)]
        )

    def gnn(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        return x

    def forward(self, x, edge_index):
        globfeat = self.global_features(x)

        x = self.gnn(x, edge_index)
        x = self.graph_aggr(x)[0]
        x = torch.cat([x, globfeat], dim=-1)
        x = self.dnn(x)
        return x
