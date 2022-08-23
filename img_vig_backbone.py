import torch
import torch.nn as nn
from timm.models.layers import DropPath
import torch.nn.functional as F
from img_utils import EdgeConv2d, MRConv2d, GraphSAGE, GINConv2d
import argparse


#----------Image encoding----------

class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.img_embedding = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            nn.ReLU(),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.img_embedding(x)
        return x


#--------------------Grapher Block--------------------------------------

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


class DenseDilatedKnnGraph(nn.Module):
    def __init__(self, k):
        super(DenseDilatedKnnGraph, self).__init__()
        self.k = k

    def pairwise_distance(self, x):
        """
        Compute pairwise distance of a point cloud.
        Args:
            x: tensor (batch_size, num_points, num_dims)
        Returns:
            pairwise distance: (batch_size, num_points, num_points)
        """
        with torch.no_grad():
            x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
            x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
            dist = x_square + x_inner + x_square.transpose(2, 1)

            return dist

    def dense_knn_matrix(self, x, k):
        """Get KNN based on the pairwise distance.
        Args:
            x: (batch_size, num_dims, num_points, 1)
            k: int
        Returns:
            nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
        """
        with torch.no_grad():
            x = x.transpose(2, 1).squeeze(-1)
            batch_size, n_points, n_dims = x.shape

            dist = self.pairwise_distance(x.detach())
            _, nn_idx = torch.topk(-dist, k=k)

            center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, self.k, 1).transpose(2, 1)
            edge_idx = torch.stack((nn_idx, center_idx), dim=0)

        return edge_idx


    def forward(self, x):
        x = F.normalize(x, p=2.0, dim=1)
        edge_idx = self.dense_knn_matrix(x, self.k)

        return edge_idx


# ------------------FNN Block---------------------------


class FFN(nn.Module):
    def __init__(self, in_features, out_features, drop_path=0.0):
        super(FFN, self).__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(in_features, in_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc(x)
        x = self.drop_path(x) + shortcut
        return x



#----------------------Graph Embedding Backbone--------------------------------------


class ImgViGBackbone(nn.Module):
    """
    Input --- Image: (Batch Size, C_dimension, Height, Weight)
    Forward --- Using Vision Graph Block (Grapher + FNN) & Nonlinear Classifier (MLP)
    Output --- Num Classes : (Batch Size, Classes)
    """
    def __init__(self, channels_dim, k, num_classes=1000, conv_class='edge', drop_path=0.0, drop_out=0.0):
        super(ImgViGBackbone, self).__init__()

        self.stem = Stem(in_dim=3, out_dim=768)

        self.vig_block = nn.Sequential(Grapher(in_channels=channels_dim, k=k, conv_class=conv_class, drop_path=drop_path),
                                       FFN(in_features=channels_dim, out_features=channels_dim, drop_path=drop_path),
                                       )

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=channels_dim, out_channels=1024, kernel_size=1, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Conv2d(1024, num_classes, 1, bias=True))

    def forward(self, inputs):
        x = self.Stem(inputs)
        pos_emb = self.pos_embed
        x = self.vig_block(x + pos_emb)
        x = F.adaptive_max_pool2d(x, 1)
        x = self.classifier(x).squeeze(-1).squeeze(-1)

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
    parser.add_argument('--backbone_select', type=str, default='img', help='Selection of ViG Backbone --feature or img')

    args = parser.parse_args()

    if args.backbone_select == 'img':
        img_tensor = torch.randn(4, 3, 224, 224)
        B, C, H, W = img_tensor.shape

        backbone = ImgViGBackbone(channels_dim=768, k=3, num_classes=1000, conv_class='edge', drop_path=0.0, drop_out=0.0)
        if torch.cuda.is_available() == True:
            img_tensor = img_tensor.cuda()
            backbone = backbone.cuda()

        imgs_pred = backbone(img_tensor)