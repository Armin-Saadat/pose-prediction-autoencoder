import argparse
import torch
import torch.nn as nn
from utils.others import set_dataloader, set_model, load_model, AverageMeter, speed2pos
from utils.metrices import ADE_c, FDE_c
import time
import sys


def parse_option():
    parser = argparse.ArgumentParser('argument for predictions')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input', type=int, default=16)
    parser.add_argument('--output', type=int, default=14)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=1000)
    parser.add_argument('--hardtanh_limit', type=int, default=100)
    parser.add_argument('--load_global_ckpt', type=str)
    parser.add_argument('--dropout_encoder', type=float, default=0)
    parser.add_argument('--dropout_pose_decoder', type=float, default=0)
    parser.add_argument('--dropout_mask_decoder', type=float, default=0)
    parser.add_argument('--n_layers', type=int, default=1)
    opt = parser.parse_args()
    opt.stride = opt.input
    opt.skip = 1
    opt.dataset_name = 'posetrack'
    opt.loader_shuffle = True
    opt.pin_memory = False
    return opt


def predict(loader, global_model):
    l1e = nn.L1Loss()
    bce = nn.BCELoss()
    start = time.time()
    avg_epoch_global_speed_loss = AverageMeter()
    avg_epoch_mask_loss = AverageMeter()
    ade_val = AverageMeter()
    fde_val = AverageMeter()
    for idx, (obs_velocities, target_velocities, obs_pose, target_pose, obs_mask, target_mask) in enumerate(loader):
        global_vel_obs = obs_velocities[:, :, :2].to(device='cuda')
        global_vel_targets = target_velocities[:, :, :2].to(device='cuda')
        global_pose_obs = obs_pose[:, :, :2].to(device='cuda')
        local_pose_obs = obs_pose[:, :, 2:].to(device='cuda')
        target_pose = target_pose.to(device='cuda')
        obs_mask = obs_mask.to(device='cuda')
        mask_target = target_mask.to(device='cuda')
        with torch.no_grad():
            vel = global_vel_preds = global_model(pose=global_pose_obs, vel=global_vel_obs)
            local_vel_preds = torch.cat((vel, vel, vel, vel, vel, vel, vel, vel, vel, vel, vel, vel, vel), 2)
            m = obs_mask[:, -1:, :]
            mask_preds = torch.cat((m, m, m, m, m, m, m, m, m, m, m, m, m, m), 1)
            global_speed_loss = l1e(global_vel_preds, global_vel_targets)
            mask_loss = bce(mask_preds, mask_target)
            avg_epoch_global_speed_loss.update(val=float(global_speed_loss), n=global_vel_targets.shape[0])
            avg_epoch_mask_loss.update(val=float(mask_loss), n=mask_target.shape[0])

            global_pose_pred = speed2pos(global_vel_preds, global_pose_obs)
            local_pose_pred = speed2pos(local_vel_preds, local_pose_obs)

            pose_pred = torch.cat((global_pose_pred, local_pose_pred), 2)
            ade_val.update(val=float(ADE_c(pose_pred, target_pose)), n=target_pose.shape[0])
            fde_val.update(val=FDE_c(pose_pred, target_pose), n=target_pose.shape[0])

    print('| speed_loss: %.2f' % avg_epoch_global_speed_loss.avg,
          '| mask_loss: %.2f' % avg_epoch_mask_loss.avg,
          '| ade_val: %.2f' % ade_val.avg, '| fde_val: %.2f' % fde_val.avg,
          '| epoch_time.avg:%.2f' % (time.time() - start))
    sys.stdout.flush()


if __name__ == '__main__':
    opt = parse_option()
    opt.model_name = 'de_predict'
    _, val_loader = set_dataloader(opt)
    opt.model_name = 'de_global'
    global_model = set_model(opt)
    if opt.load_global_ckpt is not None:
        global_model = load_model(opt, global_model, opt.load_global_ckpt)
    else:
        raise EnvironmentError
    predict(val_loader, global_model)
