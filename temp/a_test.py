import torch
from vig_tinytiny import vig_tinytiny

if __name__ == '__main__':

    inputs = torch.randn(4, 3, 224, 224).cuda()

    model = vig_tinytiny(channels_dim=768, k=9, num_classes=1000, conv_class='edge', drop_rate=0.0)
    model = model.cuda()

    preds = model(inputs)

    print('ok')