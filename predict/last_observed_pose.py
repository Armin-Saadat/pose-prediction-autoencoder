import argparse
import torch
import torch.nn as nn
from utils.others import set_dataloader, AverageMeter, speed2pos
from utils.metrices import ADE_c, FDE_c
import time
import sys


def parse_option():
    parser = argparse.ArgumentParser('argument for predictions')
    parser.add_argument('--input', type=int, default=16)
    parser.add_argument('--output', type=int, default=14)
    parser.add_argument('--batch_size', type=int, default=64)
    opt = parser.parse_args()
    opt.dataset_name = 'posetrack'
    opt.loader_shuffle = True
    opt.pin_memory = False
    return opt


def predict(loader):
    l1e = nn.L1Loss()
    bce = nn.BCELoss()
    val_s_scores = []

    start = time.time()
    avg_epoch_val_speed_loss = AverageMeter()
    avg_epoch_val_pose_loss = AverageMeter()

    ade_val = AverageMeter()
    fde_val = AverageMeter()
    for idx, (obs_s, target_s, obs_pose, target_pose, obs_mask, target_mask) in enumerate(loader):
        target_s = target_s.to(device='cuda')
        obs_pose = obs_pose.to(device='cuda')
        target_pose = target_pose.to(device='cuda')
        obs_mask = obs_mask.to(device='cuda')
        target_mask = target_mask.to(device='cuda')
        with torch.no_grad():
            batch_size = obs_pose.shape[0]
            speed_preds = torch.zeros(batch_size, 14, 28)
            m = obs_mask[:, -1, :]
            mask_preds = torch.cat((m, m, m, m, m, m, m, m, m, m, m, m, m, m), 1)

            speed_loss = l1e(speed_preds, target_s)
            mask_loss = bce(mask_preds, target_mask)

            avg_epoch_val_speed_loss.update(val=float(speed_loss))
            avg_epoch_val_pose_loss.update(val=float(mask_loss))

            preds_p = speed2pos(speed_preds, obs_pose)
            ade_val.update(val=float(ADE_c(preds_p, target_pose)))
            fde_val.update(val=float(FDE_c(preds_p, target_pose)))

    val_s_scores.append(avg_epoch_val_speed_loss.avg)
    print('| validation_speed_loss: %.2f' % avg_epoch_val_speed_loss.avg,
          '| validation_mask_loss: %.2f' % avg_epoch_val_pose_loss.avg,
          '| ade_val: %.2f' % ade_val.avg, '| fde_val: %.2f' % fde_val.avg,
          '| epoch_time.avg:%.2f' % (time.time() - start))
    sys.stdout.flush()


if __name__ == '__main__':
    opt = parse_option()
    _, val_loader = set_dataloader(opt)
    predict(val_loader)
