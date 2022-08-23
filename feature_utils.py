import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k):
        super(DenseDilated, self).__init__()
        self.k = k

    def forward(self, edge_idx):
        edge_idx = edge_idx[:, :, :, ::1]

        return edge_idx




class DenseDilatedKnnGraph(nn.Module):
    def __init__(self, k):
        super(DenseDilatedKnnGraph, self).__init__()
        self.k = k
        self._dilated = DenseDilated(k)

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
        edge_idx = self._dilated(edge_idx)

        return edge_idx


######---------------------------------------------Graph Conv----------------------------------------------------######


def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    x = x.unsqueeze(0).unsqueeze(-1)
    batch_size, num_vertices_reduced, num_dims = x.shape[:3]
    idx = idx.unsqueeze(0)
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels):
        super(EdgeConv2d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=True, groups=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            )

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value

class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels):
        super(MRConv2d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=True, groups=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)

class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.nn1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=True, groups=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

        self.nn2 = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, 1, bias=True, groups=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels):
        super(GINConv2d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=True, groups=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)

# def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
#     # activation layer
#
#     act = act.lower()
#     if act == 'relu':
#         layer = nn.ReLU(inplace)
#     elif act == 'leakyrelu':
#         layer = nn.LeakyReLU(neg_slope, inplace)
#     elif act == 'prelu':
#         layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
#     elif act == 'gelu':
#         layer = nn.GELU()
#     elif act == 'hswish':
#         layer = nn.Hardswish(inplace)
#     else:
#         raise NotImplementedError('activation layer [%s] is not found' % act)
#     return layer