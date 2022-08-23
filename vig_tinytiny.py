import torch
import torch.nn as nn
from temp.Grapher_pre_version import Grapher
from FFN import FFN
import torch.nn.functional as F



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


class vig_tinytiny(nn.Module):
    '''
    Vision Graph Model with single "vig_block: Grapher+FFN" Block
    '''
    def __init__(self, channels_dim, k, num_classes=1000, conv_class='edge', drop_rate=0.0):
        super(vig_tinytiny, self).__init__()

        self.pos_embed = nn.Parameter(torch.zeros(1, 768, 224 // 4, 224 // 4))

        self.Stem = Stem(in_dim=3, out_dim=768)

        self.vig_block = nn.Sequential(Grapher(in_channels=channels_dim, k=k, conv_class=conv_class, drop_path=0.0),
                                       FFN(in_features=channels_dim, out_features=channels_dim, drop_path=0.0),
                                       )

        self.classifier = nn.Sequential(nn.Conv2d(in_channels=channels_dim, out_channels=1024, kernel_size=1, bias=True),
                                        nn.BatchNorm2d(1024),
                                        nn.ReLU(),
                                        nn.Dropout(p=drop_rate),
                                        nn.Conv2d(1024, num_classes, 1, bias=True))

    def forward(self, inputs):
        x = self.Stem(inputs)
        pos_emb = self.pos_embed
        x = self.vig_block(x + pos_emb)
        x = F.adaptive_max_pool2d(x, 1)
        x = self.classifier(x).squeeze(-1).squeeze(-1)

        return x


    # channel = 512        # dimension
    # k = 9                  # neighbor num

    # model = vig_tinytiny(channels_dim=768, k=9, num_classes=1000, conv_class='edge', drop_rate=0.0)
    # model = model.cuda()

    # preds = model(inputs)

    # print('ok')




    ###--------------------New Vision Graph-------------------###









