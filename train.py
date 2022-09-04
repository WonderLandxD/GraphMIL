import argparse
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from util import AverageMeter
from tqdm import tqdm
import torch.utils.data as Dataloader
from torch.utils.data import Dataset
from feature_vig_backbone import GraphMILBackbone
from random import sample


class SYSDataset(Dataset):
    def __init__(self, feat_dir=None):
        self.feat_list = glob.glob(os.path.join(feat_dir, '*/*.pt'))

    def __getitem__(self, item):
        _feat = self.feat_list[item]
        feat = torch.load(_feat)
        data_class = _feat.split('/')[-2]
        if data_class == 'LUAD':
            label = 0
        elif data_class == 'LUSC':
            label = 1
        else:
            label = 2

        return feat, label

    def __len__(self):
        return(len(self.feat_list))

def model_train(model, train_loader, criterion, optimizer, epoch):
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    end = time.time()

    for batch_idx, (input, label) in enumerate(tqdm(train_loader, disable=False)):

        optimizer.zero_grad()
        input = input.cuda()
        label = label.cuda()
        nlist = range(0, input.shape[1])
        if input.shape[1] > args.upper_bound:
            sample_list = sample(nlist, args.upper_bound)
            sample_index = torch.tensor(sample_list, dtype=torch.int).cuda()
            input = input.index_select(1, sample_index)

        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        batch_size = label.size(0)
        losses.update(loss.item(), batch_size)
        pred = torch.argmax(output, dim=1)

        acc.update(torch.sum(label == pred).item() / batch_size, batch_size)

        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print statistics and write summary every N batch
        if (batch_idx + 1) % len(train_loader) == 0:
            print('{}'.format("GraphMIL"))
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, batch_idx + 1, len(train_loader), batch_time=batch_time, loss=losses,
                acc=acc))

    return losses.avg, acc.avg

def model_val(model, val_loader, criterion, epoch):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    with torch.no_grad():
        end = time.time()

        for batch_idx, (input, label) in enumerate(tqdm(val_loader, disable=False)):
            input = input.cuda()
            label = label.cuda()
            nlist = range(0, input.shape[1])

            if input.shape[1] > args.upper_bound:
                sample_list = sample(nlist, args.upper_bound)
                sample_index = torch.tensor(sample_list, dtype=torch.int).cuda()
                input = input.index_select(1, sample_index)

            output = model(input)
            loss = criterion(output, label)

            batch_size = label.size(0)
            losses.update(loss.item(), batch_size)
            pred = torch.argmax(output, dim=1)

            acc.update(torch.sum(label == pred).item() / batch_size, batch_size)

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # print statistics and write summary every N batch
            if (batch_idx + 1) % len(val_loader) == 0:
                print('{}'.format('GraphMIL'))
                print('Val: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, batch_idx + 1, len(val_loader), batch_time=batch_time, loss=losses,
                    acc=acc))

    return losses.avg, acc.avg

def parse_args():

    parser = argparse.ArgumentParser('Argument for training')

    parser.add_argument('--gpu', type=str, default='4', help='GPU Setting, default=0')
    parser.add_argument('--num_epoch', type=int, default=1000, help='NUMBER OF EPOCH, default=100')
    parser.add_argument('--lr', default=0.0001, type=float, help='LEARNING RATE, default=0.0001')
    parser.add_argument('--dataset', default='SYS', type=str, help='DATASET, default=SYS')
    parser.add_argument('--upper_bound', type=int, default=8000, help='SAMPLER NUMBER, default=8000')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.dataset == 'SYS':

        print('-----------------------SYS Datasets GraphMIL Training (LUAD, LUSC, Non-neoplasticDiseases)-----------------------')

        DATA_ROOT = '/pub/data/chizm/SYSFL/SYS_ThreeClasses_Features'

        full_dataset = SYSDataset(DATA_ROOT)

        train_size = int(0.7 * len(full_dataset))
        valid_size = len(full_dataset) - train_size

        train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])

        train_loader = Dataloader.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
        valid_loader = Dataloader.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

        ### Save log ###
        # save_log = './SYS_Results/Cross_Validation'
        _save_folder = './SYS_Results/DemoTrainVal'
        for i in range(1, 100):
            save_folder = '{}_{}'.format(_save_folder, i)
            if os.path.exists(save_folder):
                i += 1
            else:
                os.makedirs(save_folder)
                break

        n_data = len(train_dataset)
        NUM_CLASSES = len(os.listdir(DATA_ROOT))

        print('---number of training samples: {}---'.format(n_data))
        print('---number of training samples: {}---'.format(n_data),
              file=open(os.path.join(save_folder, 'datasetting.txt'), 'w'))
        n_data = len(valid_dataset)
        print('---number of validation samples: {}---'.format(n_data))
        print('---number of validation samples: {}---'.format(n_data),
              file=open(os.path.join(save_folder, 'datasetting.txt'), 'a'))

        model = GraphMILBackbone(in_features=2048, num_classes=NUM_CLASSES, conv_class='edge',
                                 drop_path=0.0, drop_out=0.0)

        print('---Using {} Backbone---'.format('GraphMIL'))

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)


        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Training Model
        start_epoch = 1
        pre_best_val_acc = float('-inf')

        with open(os.path.join(save_folder, 'train_val_results.csv'), 'w') as f:
            f.write('epoch, train_loss, train_acc, val_loss, val_acc\n')

        EPOCH = args.num_epoch

        for epoch in range(start_epoch, EPOCH + 1):

            print('==> training...')

            time_start = time.time()

            train_losses, train_acc = model_train(model, train_loader, criterion, optimizer, epoch)
            print('Epoch time: {:.2f} s.'.format(time.time() - time_start))

            print('==> validation...')
            val_losses, val_acc = model_val(model, valid_loader, criterion, epoch)

            # log results
            with open(os.path.join(save_folder, 'train_val_results.csv'), 'a') as f:
                f.write('%03d,%0.6f,%0.6f,%0.6f,%0.6f,\n' % (epoch, train_losses, train_acc, val_losses, val_acc))

            if (val_acc > pre_best_val_acc):
                print('==> best acc Saving...')
                torch.save(model.state_dict(), '{}/acc_best_model.pth'.format(save_folder))
                pre_best_val_acc = val_acc

            if epoch % 10 == 0:
                print('==> routine Saving...')
                torch.save(model.state_dict(), '{}/epoch_{}_model.pth'.format(save_folder, epoch))










