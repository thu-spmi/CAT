'''
Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang, Keyu An
Apache 2.0.
This script shows how to excute CTC-CRF neural network training with PyTorch.
'''
from ctc_crf import CTC_CRF_LOSS
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import timeit
import os
import sys
import argparse
import json
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from model import BLSTM, LSTM, VGGBLSTM, VGGLSTM, LSTMrowCONV, TDNN_LSTM, BLSTMN
from dataset import SpeechDataset, SpeechDatasetMem, SpeechDatasetPickle, SpeechDatasetMemPickle, PadCollate
import ctc_crf_base
from torch.utils.tensorboard import SummaryWriter
from utils import save_ckpt, optimizer_to

TARGET_GPUS = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(",")))]
gpus = torch.IntTensor(TARGET_GPUS)
ctc_crf_base.init_env('data/den_meta/den_lm.fst', gpus)


class Model(nn.Module):
    def __init__(self, net, idim, hdim, K, n_layers, dropout, lamb):
        super(Model, self).__init__()
        self.net = eval(net)(idim, hdim, n_layers, dropout=dropout)
        if net in ['BLSTM', 'BLSTMN', 'VGGBLSTM']:
            self.linear = nn.Linear(hdim * 2, K)
        else:
            self.linear = nn.Linear(hdim, K)
        self.loss_fn = CTC_CRF_LOSS(lamb=lamb)

    def forward(self, logits, labels_padded, input_lengths, label_lengths):
        # rearrange by input_lengths
        input_lengths, indices = torch.sort(input_lengths, descending=True)
        assert indices.dim() == 1, "input_lengths should have only 1 dim"
        logits = torch.index_select(logits, 0, indices)
        labels_padded = torch.index_select(labels_padded, 0, indices)
        label_lengths = torch.index_select(label_lengths, 0, indices)

        labels_padded = labels_padded.cpu()
        input_lengths = input_lengths.cpu()
        label_lengths = label_lengths.cpu()

        label_list = [
            labels_padded[i, :x] for i, x in enumerate(label_lengths)
        ]
        labels = torch.cat(label_list)
        netout, input_lengths = self.net(logits, input_lengths)
        input_lengths = input_lengths.to(dtype=torch.int32)
        netout = self.linear(netout)
        netout = F.log_softmax(netout, dim=2)
        loss = self.loss_fn(netout, labels, input_lengths, label_lengths)
        return loss


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    parser = argparse.ArgumentParser(description="recognition argument")
    parser.add_argument("dir", default="models")
    parser.add_argument(
        "--arch",
        choices=[
            'BLSTM', 'LSTM', 'VGGBLSTM', 'VGGLSTM', 'LSTMrowCONV', 'TDNN_LSTM',
            'BLSTMN', 'TDNN_downsample'
        ],
        default='BLSTM')
    parser.add_argument("--min_epoch", type=int, default=5)
    parser.add_argument("--output_unit", type=int)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--hdim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--feature_size", type=int, default=120)
    parser.add_argument("--data_path")
    parser.add_argument("--lr", type=float,default=0.001)
    parser.add_argument("--stop_lr", type=float,default=0.00001)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clip_grad", action="store_true")
    parser.add_argument("--pkl", action="store_true")
    parser.add_argument("--pretrained_model_path")
    args = parser.parse_args()

    os.makedirs(args.dir + '/board', exist_ok=True)
    writer = SummaryWriter(args.dir +'/board')
    # save configuration
    with open(args.dir + '/config.json', "w") as fout:
        config = {
            "arch": args.arch,
            "output_unit": args.output_unit,
            "hdim": args.hdim,
            "layers": args.layers,
            "dropout": args.dropout,
            "feature_size": args.feature_size,
        }
        json.dump(config, fout)

    epoch = 0   
    model = Model(args.arch, args.feature_size, args.hdim, args.output_unit,
                  args.layers, args.dropout, args.lamb)
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    min_cv_loss = np.inf
    
    device = torch.device("cuda:0")

    if args.resume:
        print("resume from {}".format(args.pretrained_model_path))
        pretrained_dict = torch.load(args.pretrained_model_path)
        epoch = pretrained_dict['epoch']
        model.load_state_dict(pretrained_dict['model'])
        lr = pretrained_dict['lr']
        optimizer.load_state_dict(pretrained_dict['optimizer'])
        optimizer_to(optimizer, device)
        min_cv_loss = pretrained_dict['cv_loss']

    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)
       
    if args.pkl:
        tr_dataset = SpeechDatasetMemPickle(args.data_path + "/tr.pkl") 
    else:
        tr_dataset = SpeechDatasetMem(args.data_path + "/tr.hdf5")

    tr_dataloader = DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
        collate_fn=PadCollate())

    if args.pkl:
        cv_dataset = SpeechDatasetMemPickle(args.data_path + "/cv.pkl") 
    else:
        cv_dataset = SpeechDatasetMem(args.data_path + "/cv.hdf5")

    cv_dataloader = DataLoader(
        cv_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        collate_fn=PadCollate())

    prev_t = 0
    model.train()
    while True:

        # training stage
        epoch += 1
        for i, minibatch in enumerate(tr_dataloader):
            print("training epoch: {}, step: {}".format(epoch, i))
            logits, input_lengths, labels_padded, label_lengths, path_weights = minibatch

            sys.stdout.flush()
            model.zero_grad()
            optimizer.zero_grad()

            loss = model(logits, labels_padded, input_lengths, label_lengths)
            partial_loss = torch.mean(loss.cpu())
            weight = torch.mean(path_weights)
            real_loss = partial_loss - weight

            loss.backward(loss.new_ones(loss.shape[0]))
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            t2 = timeit.default_timer()
            writer.add_scalar('training loss',
                            real_loss.item(),
                            (epoch-1) * len(tr_dataloader) + i)
            print("time: {}, tr_real_loss: {}, lr: {}".format(t2 - prev_t, real_loss.item(), optimizer.param_groups[0]['lr']))
            prev_t = t2
            
        # cv stage
        model.eval()
        cv_losses_sum = []
        count = 0

        for i, minibatch in enumerate(cv_dataloader):
            print("cv epoch: {}, step: {}".format(epoch, i))
            logits, input_lengths, labels_padded, label_lengths, path_weights = minibatch

            loss = model(logits, labels_padded, input_lengths, label_lengths)
            loss_size = loss.size(0)
            count = count + loss_size
            partial_loss = torch.mean(loss.cpu())
            weight = torch.mean(path_weights)
            real_loss = partial_loss - weight
            real_loss_sum = real_loss * loss_size
            cv_losses_sum.append(real_loss_sum.item())
            print("cv_real_loss: {}".format(real_loss.item()))

        cv_loss = np.sum(np.asarray(cv_losses_sum)) / count
        print("mean_cv_loss: {}".format(cv_loss))
        writer.add_scalar('mean_cv_loss',cv_loss,epoch)
        
        # save model
        save_ckpt({
            'epoch': epoch,
            'model': model.module.state_dict(),
            'lr': lr,
            'optimizer': optimizer.state_dict(),
            'cv_loss': cv_loss
            }, epoch < args.min_epoch or cv_loss <= min_cv_loss, args.dir, "model.epoch.{}".format(epoch))
        
        if epoch < args.min_epoch or cv_loss <= min_cv_loss:
            min_cv_loss = cv_loss                
        else:
            print(
                "cv loss does not improve, decay the learning rate from {} to {}"
                .format(lr, lr / 10.0))
            adjust_lr(optimizer, lr / 10.0)
            lr = lr / 10.0
            if (lr < args.stop_lr):
                print("learning rate is too small, finish training")
                break

        model.train()

    ctc_crf_base.release_env(gpus)


if __name__ == "__main__":
    train()
