import torch
import torch.nn as nn
from utils import DenseDilatedKnnGraph
from utils import EdgeConv2d, MRConv2d, GraphSAGE, GINConv2d
from timm.models.layers import DropPath

class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv_class):
        super(GraphConv2d, self).__init__()
        if conv_class == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels)
        elif conv_class == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels)
        elif conv_class == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels)
        elif conv_class == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv_class))

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class Dynamic_GraphConv2d(GraphConv2d):
    def __init__(self, in_channels, out_channels, k, conv_class):
        super(Dynamic_GraphConv2d, self).__init__(in_channels, out_channels, conv_class)
        self.k = k

        self.dilated_knn_graph = DenseDilatedKnnGraph(k)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_idx = self.dilated_knn_graph(x)
        x = super(Dynamic_GraphConv2d, self).forward(x, edge_idx)

        return x.reshape(B, -1, H, W).contiguous()




class Grapher(nn.Module):
    def __init__(self, in_channels, k, conv_class, drop_path=0.0):
        super(Grapher, self).__init__()
        self.fc1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(in_channels),
                                 )
        self.graph_conv = Dynamic_GraphConv2d(in_channels, in_channels * 2, k, conv_class)
        self.fc2 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(in_channels),
                                 )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.relative_pos = None


    def forward(self, x):
        _temp = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + _temp

        return x


