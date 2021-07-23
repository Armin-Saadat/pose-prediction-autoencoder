import argparse
import torch
import torch.nn as nn
from utils.others import set_dataloader, AverageMeter, speed2pos
from utils.metrices import ADE_c, FDE_c, mask_accuracy
import time
import sys


def parse_option():
    parser = argparse.ArgumentParser('argument for predictions')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input', type=int, default=16)
    parser.add_argument('--output', type=int, default=14)
    parser.add_argument('--batch_size', type=int, default=64)
    opt = parser.parse_args()
    opt.stride = opt.input
    opt.skip = 1
    opt.dataset_name = 'posetrack'
    opt.loader_shuffle = True
    opt.pin_memory = False
    opt.model_name = 'lstm_vel'
    return opt


def predict(loader):
    l1e = nn.L1Loss()
    bce = nn.BCELoss()
    start = time.time()
    avg_epoch_speed_loss = AverageMeter()
    avg_epoch_mask_loss = AverageMeter()
    avg_epoch_mask_acc = AverageMeter()
    ade_val = AverageMeter()
    fde_val = AverageMeter()
    for idx, (obs_s, target_s, obs_pose, target_pose, obs_mask, target_mask) in enumerate(loader):
        obs_s = obs_s.to(device='cuda')
        target_s = target_s.to(device='cuda')
        obs_pose = obs_pose.to(device='cuda')
        target_pose = target_pose.to(device='cuda')
        obs_mask = obs_mask.to(device='cuda')
        target_mask = target_mask.to(device='cuda')
        with torch.no_grad():
            s, m = obs_s[:, -1:, :], obs_mask[:, -1:, :]
            speed_preds = torch.cat((s, s, s, s, s, s, s, s, s, s, s, s, s, s), 1)
            mask_preds = torch.cat((m, m, m, m, m, m, m, m, m, m, m, m, m, m), 1)

            speed_loss = l1e(speed_preds, target_s)
            mask_loss = bce(mask_preds, target_mask)
            mask_acc = mask_accuracy(mask_preds, target_mask)

            avg_epoch_speed_loss.update(val=float(speed_loss), n=target_s.shape[0])
            avg_epoch_mask_loss.update(val=float(mask_loss), n=target_mask.shape[0])
            avg_epoch_mask_acc.update(val=float(mask_acc), n=target_mask.shape[0])

            preds_p = speed2pos(speed_preds, obs_pose)
            ade_val.update(val=float(ADE_c(preds_p, target_pose)), n=target_pose.shape[0])
            fde_val.update(val=float(FDE_c(preds_p, target_pose)), n=target_pose.shape[0])

    print('| speed_loss: %.2f' % avg_epoch_speed_loss.avg,
          '| mask_loss: %.2f' % avg_epoch_mask_loss.avg,
          '| mask_acc: %.2f' % avg_epoch_mask_acc.avg,
          '| ade_val: %.2f' % ade_val.avg, '| fde_val: %.2f' % fde_val.avg,
          '| epoch_time.avg:%.2f' % (time.time() - start))
    sys.stdout.flush()


if __name__ == '__main__':
    opt = parse_option()
    _, val_loader = set_dataloader(opt)
    predict(val_loader)
