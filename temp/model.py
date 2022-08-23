import torch
import torch.nn as nn
from Grapher_pre_version import Grapher
from FFN import FFN

##############################
#    Basic layers
##############################
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),         # O:112
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


if __name__ == '__main__':











    #opt
    channels = [48, 96, 240, 384]         # dimension
    k = 9                  # neighbor num

    patch_embedding = Stem(img_size=224, in_dim=3, out_dim=channels[0], act='relu')
    img_feature = patch_embedding(img_tensor)        # size:(4, channels, 56, 56)

    grapher_module = Grapher(in_channels=channels[0], k=k, conv_class='edge', drop_path=0.0)
    img_grapher_feature = grapher_module(img_feature)

    ffn_module = FFN(in_features=channels[0], out_features=channels[0], drop_path=0.0)
    img_ffn_feature = ffn_module(img_grapher_feature)

    print('ok')