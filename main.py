import os
import random
import sys
import torch.optim
import numpy as np
import argparse

import yaml
from torch.utils.data import DataLoader

from data_gen.human_36m import h36m
from module.Model import ModelLayer
from utils import runtime
from utils.runtime import TrainData, Trainer
from utils.runtime import test, save_ckpt
from utils.batch_sample import get_batch_srnn


def run():
    # model loading
    print(">>> loading model")
    model = ModelLayer(args.key_point, args.seq_len, args.pred_len, args.d_ff, args.d_emb, args.d_heads,
                       args.drop_out, args.linear_n, args.nonlinear_n, args.patch_size).cuda()

    print(">>> total params: {:.3f} M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    # data loading
    dataset = h36m(data_dir=args.data)

    if args.mode == 'train':

        print(">>> loading test data")
        test_set = dataset.load_data(dataset.eval_subject, args.act)
        test_gen = {}
        for act in dataset.define_acts(args.act):
            test_gen[act] = get_batch_srnn(test_set, act, args.seq_len, args.pred_len, args.key_point, args.manner)

        print(">>> loading train data")
        train_set = dataset.load_data(dataset.train_subject, args.act)
        train_gen = TrainData(train_set, args.seq_len, args.pred_len, sample_start=16, data_flip=args.aug)
        loader_train = DataLoader(train_gen, batch_size=args.batch_size, shuffle=True)
        work_dir, log = runtime.exp_create(args.exp, h36m.exp, args.cfg)
        print(f">>> start to train model")
        train = Trainer(model.parameters(), args.lr, args.lr_step, args.gamma, args.weight_decay, args.loss_type)

        for epoch in range(1, args.epochs + 1):
            lr, loss = train.epoch(model, loader_train)
            print(f">>> train Epoch: {epoch} Lr: {lr:.7f} loss: {loss:.7f} Manner: {args.manner}")
            if epoch % args.eval_interval == 0:
                if args.pred_len > 12:
                    eval_frame = [(1, 80), (3, 160), (7, 320), (9, 400), (13, 560), (17, 720), (21, 880), (24, 1000)]
                else:
                    eval_frame = [(1, 80), (3, 160), (7, 320), (9, 400)]

                print(f">>> eval Epoch: {epoch} Lr: {lr:.7f} loss: {loss:.7f} Manner: {args.manner}", file=log, flush=True)
                avg_angle, avg_avg_angle, eval_msg = test(test_gen, model, eval_frame, dataset.define_acts(args.act))
                print(eval_msg)
                print(eval_msg, file=log, flush=True)
                state = {'epoch': epoch,
                         'loss': loss,
                         'state_dict': model.state_dict(),
                         'optimizer': train.optimizer.state_dict()}
                save_ckpt(work_dir, h36m.exp, state, epoch, avg_avg_angle, args.keep)

    elif args.mode == 'eval':

        print(">>> loading test data")
        test_set = dataset.load_data(dataset.eval_subject, args.act)
        test_gen = {}
        for act in dataset.define_acts(args.act):
            test_gen[act] = get_batch_srnn(test_set, act, args.seq_len, args.pred_len, args.key_point, args.manner)

        print(f">>> start to test model")
        if args.pred_len > 12:
            eval_frame = [(1, 80), (3, 160), (7, 320), (9, 400), (13, 560), (17, 720), (21, 880), (24, 1000)]
        else:
            eval_frame = [(1, 80), (3, 160), (7, 320), (9, 400)]

        print(">>> loading ckpt from '{}'".format(args.ckpt))
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['state_dict'])
        print(">>> ckpt loaded (epoch: {} | loss: {})". format(ckpt['epoch'], ckpt['loss']))
        avg_angle, avg_avg_angle, eval_msg = test(test_gen, model, eval_frame, dataset.define_acts(args.act))
        print(eval_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train && eval && infer options')
    parser.add_argument('--mode', type=str, default='train', help='train or eval or viz')
    parser.add_argument('--data', type=str, default='./data/h36m', help='path to H36M dataset')
    parser.add_argument('--seed', type=int, default=1234567890, help='random seed')
    parser.add_argument('--cfg', type=str, default='cfg/h36m.yml', help='path to the configuration in .yml')
    parser.add_argument('--ckpt', type=str, default='./exp/test.pt', help='path to ckpt')
    parser.add_argument('--exp', type=str, default='./exp', help='dir to release experiment')
    parser.add_argument('--manner', type=str, default='8', help='all or 256 or 8')
    parser.add_argument('--viz', type=str, default='./demo', help='viz save path')
    parser.add_argument('--act', type=str, default='all', help='eval action')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        yml_arg = yaml.load(f, Loader=yaml.FullLoader)

    parser.set_defaults(**yml_arg)
    args = parser.parse_args()
    run()
