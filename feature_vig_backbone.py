import torch
import torch.nn as nn
from feature_utils import EdgeConv2d, MRConv2d, GraphSAGE, GINConv2d
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath
import argparse


#--------------------Grapher Block--------------------------------------

class Grapher(nn.Module):
    def __init__(self, in_features, k, conv_class, drop_path=0.0):
        super(Grapher, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_features=in_features, out_features=in_features, bias=True),
                                 nn.ReLU(),
                                 )

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


#----------------------Graph Embedding Backbone--------------------------------------


class GraphMILBackbone(nn.Module):
    """
    Input --- Feature: (N_vertices, C_dimension)
    Forward --- Using Vision Graph Block (Grapher + FNN) & Nonlinear Classifier (MLP)
    Output --- Num Classes : (N_vertices, Num Classes) / (1, Num Classes)
    """
    def __init__(self, in_features, k, num_classes=2, conv_class='edge', drop_path=0.0, drop_out=0.0):
        super(GraphMILBackbone, self).__init__()

        self.blocks = [3, 3, 3, 3]
        self.channels = [512, 640, 768, 1024]
        self.knns = [9, 12, 15, 18]
        self.backbone = nn.ModuleList([])
        #-----前n-1层block-----
        for i in range(len(self.blocks) - 1):
            for j in range(self.blocks[i] - 1):
                self.backbone += [nn.Sequential(Grapher(in_features=self.channels[i], k=self.knns[i], conv_class=conv_class, drop_path=drop_path),
                                                FFN(in_features=self.channels[i], out_features=self.channels[i], drop_path=drop_path)
                                                )
                                  ]
            self.backbone += [nn.Sequential(Grapher(in_features=self.channels[i], k=self.knns[i], conv_class=conv_class, drop_path=drop_path),
                                            FFN(in_features=self.channels[i], out_features=self.channels[i+1], drop_path=drop_path)
                                            )
                              ]
        #-----最后一层block-----
        for j in range(self.blocks[-1]):
            self.backbone += [nn.Sequential(Grapher(in_features=self.channels[-1], k=self.knns[-1], conv_class=conv_class, drop_path=drop_path),
                                            FFN(in_features=self.channels[-1], out_features=self.channels[-1], drop_path=drop_path)
                                            )
                             ]

        self.backbone = nn.Sequential(*self.backbone)

        # self.vig_block = nn.Sequential(Grapher(in_features, k, conv_class, drop_path=drop_path),
        #                                FFN(in_features=in_features, out_features=in_features, drop_path=drop_path),
        #                                )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.channels[-1], out_features=self.channels[-1] // 2, bias=True),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(self.channels[-1] // 2, num_classes, bias=True)
        )

    def forward(self, inputs):
        #需修改
        # pos_emb = self.pos_embed
        x = self.backbone(inputs)
        # embedding_pooling ---> AvgPooling         (这里有两种思路：一种embedding_pooling：max，avg，att；一种Class Token，基于TransMIL)
        x_sum = torch.sum(x, dim=0)
        x = torch.div(x_sum, int(x[0]))

        x = self.classifier(x)

        return x


#-------------------------Main Function------------------------------------


if __name__ == '__main__':
    # FeatEncoder = timm.create_model('resnet18', pretrained=True, features_only=True)
    # FeatEncoder = FeatEncoder.cuda()
    #
    # WSI_tensor = torch.randn(1, 128, 3, 224, 224).cuda()
    # WSI_tensor = WSI_tensor.squeeze(0)  # (1000, 3, 224, 224)

    # features_list = FeatEncoder(WSI_tensor)
    # feature_maps = features_list[-1]
    # feature_vectors = F.adaptive_avg_pool2d(feature_maps, 1)    # (N, C, 1, 1)
    # feature_vectors = feature_vectors.squeeze(-1).squeeze(-1)   # (N, C)

    parser = argparse.ArgumentParser(description='ViG Backbone Demo')
    parser.add_argument('--backbone_select', type=str, default='feature', help='Selection of ViG Backbone --feature or img')

    args = parser.parse_args()

    if args.backbone_select == 'feature':

        feature_tensor = torch.randn(8, 512)
        N, C = feature_tensor.shape
        in_features = feature_tensor.shape[1]


        MIL_backbones = GraphMILBackbone(in_features=in_features, k=3, num_classes=2, conv_class='edge', drop_path=0.0, drop_out=0.0)
        if torch.cuda.is_available() == True:
            feature_tensor = feature_tensor.cuda()
            MIL_backbones = MIL_backbones.cuda()

        features_pred = MIL_backbones(feature_tensor)
