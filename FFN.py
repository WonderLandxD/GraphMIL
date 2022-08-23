import torch.nn as nn
from timm.models.layers import DropPath


##############################
#    Basic layers
##############################
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
