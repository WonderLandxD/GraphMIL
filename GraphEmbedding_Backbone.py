import torch
import torch.nn as nn
from utils import EdgeConv2d, MRConv2d, GraphSAGE, GINConv2d
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath


#--------------------Grapher Block--------------------------------------

class Grapher(nn.Module):
    def __init__(self, in_features, k, conv_class, drop_path=0.0):
        super(Grapher, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_features=in_features, out_features=in_features, bias=True),
                                 nn.ReLU(),
                                 )
        # self.graph_conv = Dynamic_GraphConv2d(in_features, in_features*2, k, conv_class)

        self.graph_conv = Dynamic_GraphConv2d(in_features, in_features * 2, k, conv_class)

        self.fc2 = nn.Sequential(nn.Linear(in_features=in_features * 2, out_features=in_features, bias=True),
                                 nn.ReLU(),
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        _temp = x
        x = self.fc1(x)  # (N, C) -> (N, C)
        x = self.graph_conv(x)  # (N, C) -> (N, 2C)
        x = self.fc2(x)  # (N, 2C) -> (N, C)
        x = self.drop_path(x) + _temp

        return x


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
    def __init__(self, in_features, out_features, k, conv_class):
        super(Dynamic_GraphConv2d, self).__init__(in_features, out_features, conv_class)
        self.k = k

        self.dilated_knn_graph = DenseDilatedKnnGraph(k)

    def forward(self, x):
        N, C = x.shape
        edge_idx = self.dilated_knn_graph(x)  # (2, N, 9)
        x = super(Dynamic_GraphConv2d, self).forward(x, edge_idx)
        x = x.squeeze(0).squeeze(-1)

        return x.transpose(1, 0).contiguous()


class DenseDilatedKnnGraph(nn.Module):
    def __init__(self, k):
        super(DenseDilatedKnnGraph, self).__init__()
        self.k = k

    def pairwise_distance(self, x):
        with torch.no_grad():
            x_inner = -2 * torch.matmul(x, x.transpose(1, 0))
            x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
            dist = x_square + x_inner + x_square.transpose(1, 0)

            return dist

    def dense_knn_matrix(self, x, k):
        with torch.no_grad():
            n_points, n_dims = x.shape

            dist = self.pairwise_distance(x.detach())
            _, nn_idx = torch.topk(-dist, k=k)

            center_idx = torch.arange(0, n_points, device=x.device).repeat(self.k, 1).transpose(1, 0)
            edge_idx = torch.stack((nn_idx, center_idx), dim=0)

        return edge_idx

    def forward(self, x):
        x = F.normalize(x, p=2.0, dim=0)
        edge_idx = self.dense_knn_matrix(x, self.k)

        return edge_idx


#------------------FNN Block---------------------------


class FFN(nn.Module):
    def __init__(self, in_features, out_features, drop_path=0.0):
        super(FFN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features, bias=True),
            nn.ReLU(),
            nn.Linear(in_features, out_features, bias=True),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc(x)                            # (N, C)
        x = self.drop_path(x) + shortcut
        return x


#----------------------Graph Embedding Backbone--------------------------------------------------------------------------------

class Backbone(nn.Module):
    '''
    Vision Graph Model with single "graph embedding block: Grapher+FFN" Block
    '''
    def __init__(self, in_features, k, num_classes=2, conv_class='edge', drop_rate=0.0):
        super(Backbone, self).__init__()

        self.vig_block = nn.Sequential(Grapher(in_features, k, conv_class, drop_path=drop_rate),
                                       FFN(in_features=in_features, out_features=in_features, drop_path=drop_rate),
                                       )

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=1024, kernel_size=1, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Conv2d(1024, num_classes, 1, bias=True))

    def forward(self, inputs):
        #需修改
        x = self.Stem(inputs)
        pos_emb = self.pos_embed
        x = self.vig_block(x + pos_emb)
        x = F.adaptive_max_pool2d(x, 1)
        x = self.classifier(x).squeeze(-1).squeeze(-1)

        return x


if __name__ == '__main__':
    FeatEncoder = timm.create_model('resnet18', pretrained=True, features_only=True)
    FeatEncoder = FeatEncoder.cuda()

    WSI_tensor = torch.randn(1, 128, 3, 224, 224).cuda()
    WSI_tensor = WSI_tensor.squeeze(0)  # (1000, 3, 224, 224)

    features_list = FeatEncoder(WSI_tensor)
    feature_maps = features_list[-1]
    feature_vectors = F.adaptive_avg_pool2d(feature_maps, 1)    # (N, C, 1, 1)
    feature_vectors = feature_vectors.squeeze(-1).squeeze(-1)   # (N, C)

    N, C = feature_vectors.shape

    grapher = Grapher(in_features=C, k=9, conv_class='edge', drop_path=0.0)
    grapher = grapher.cuda()

    graph_features = grapher(feature_vectors)