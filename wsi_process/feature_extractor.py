import glob
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torch.utils.data as Dataloader
from torchvision.transforms import transforms
import timm
import argparse



class files_to_bag_Dataset(Dataset):
    def __init__(self, tiles_dir):
        self.tiles_path_list = glob.glob(os.path.join(tiles_dir, '*.png'))
        self.transform = transforms.Compose([transforms.Resize([224, 224]),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                             ])

    def __getitem__(self, item):
        _tile = Image.open(self.tiles_path_list[item]).convert('RGB')
        tile = self.transform(_tile)

        return tile

    def __len__(self):
        return len(self.tiles_path_list)


def parse_args():

    parser = argparse.ArgumentParser('Argument for training')

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--tiles_dir', type=str, default='D:\LocalPC_PythonProject\demo\\tiles')
    parser.add_argument('--feat_dir', type=str, default='D:\LocalPC_PythonProject\demo')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--feat_backbone', type=str, default='resnet50')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)

    bag_list = os.listdir(args.tiles_dir)

    print('loading model {} checkpoint'.format(args.feat_backbone))
    model = timm.create_model(args.feat_backbone, pretrained=True, num_classes=0)   # create with no classifier (pooled)
    model = model.cuda()

    # Multi-GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.eval()
    with torch.no_grad():
        i = 1
        for bag in bag_list:
            files_path = os.path.join(args.tiles_dir, bag)

            files_to_bag_dataset = files_to_bag_Dataset(files_path)

            pth_loader = Dataloader.DataLoader(files_to_bag_dataset, batch_size=args.batch_size)

            # features_box = torch.empty(len(files_to_bag_dataset), 2048).cuda()   #resnet50
            print('Extract Feature Progress: {:.2%}, {}/{} \t processing {}'.format(i / len(bag_list), i, len(bag_list),
                                                                    bag))
            for batch_idx, tiles in enumerate(tqdm(pth_loader, disable=False)):
                tiles = tiles.cuda()
                features = model(tiles)
                if batch_idx == 0:
                    features_box = features
                    continue
                else:
                    features_box = torch.cat((features_box, features), dim=0)

            torch.save(features_box, os.path.join(args.feat_dir, 'pt_files', bag+'.pt'))
            print('Tiles of slide {} has transferred to feature embeddings'.format(bag))
            i += 1



