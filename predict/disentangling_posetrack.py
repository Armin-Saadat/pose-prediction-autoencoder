import argparse
import torch
import torch.nn as nn
from utils import set_dataloader, set_model, load_model, AverageMeter, speed2pos, ADE_c, FDE_c
import time
import sys


def parse_option():
    parser = argparse.ArgumentParser('argument for predictions')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input', type=int, default=16)
    parser.add_argument('--output', type=int, default=14)
    parser.add_argument('--hidden_size', type=int, default=1000)
    parser.add_argument('--hardtanh_limit', type=int, default=100)
    parser.add_argument('--load_ckpt', type=str)

    opt = parser.parse_args()
    opt.stride = opt.input
    opt.skip = 1
    opt.dataset_name = 'posetrack'
    return opt


def predict(loader, global_model, local_model):
    l1e = nn.L1Loss()
    bce = nn.BCELoss()
    val_s_scores = []

    start = time.time()
    avg_epoch_val_speed_loss = AverageMeter()
    avg_epoch_val_pose_loss = AverageMeter()

    ade_val = AverageMeter()
    fde_val = AverageMeter()
    for idx, (obs_velocities, target_velocities, obs_pose, target_pose, obs_mask, target_mask) in loader:
        global_obs_vel = obs_velocities[:, :, 2].to(device='cuda')
        local_obs_vel = obs_velocities[:, :, 2:]
        global_target_velocities = target_velocities[:, :, 2]
        local_target_velocities = target_velocities[:, :, 2:]
        global_obs_pose = obs_pose[:, :, 2].to(device='cuda')
        local_obs_pose = obs_pose[:, :, 2].to(device='cuda')
        obs_mask = obs_mask.to(device='cuda')
        target_mask = target_mask.to(device='cuda')
        with torch.no_grad():
            (global_vel_preds, _) = global_model(pose=global_obs_pose, vel=global_obs_vel)
            (local_vel_preds, mask_preds) = local_model(pose=local_obs_pose, vel=local_obs_vel, mask=obs_mask)
            local_speed_loss = l1e(local_vel_preds, local_target_velocities)
            global_speed_loss = l1e(global_vel_preds, global_target_velocities)
            mask_loss = bce(mask_preds, target_mask)

            # avg_epoch_val_speed_loss.update(val=float(speed_loss))

            global_pose_pred = speed2pos(global_vel_preds, global_obs_pose)
            local_pose_pred = speed2pos(local_vel_preds, local_obs_pose)
            # now we have to make a prediction
            pred_pose = regenerate_entire_pose(global_pose_pred, local_pose_pred)

            ade_val.update(val=float(ADE_c(pred_pose, target_pose)))
            fde_val.update(val=float(FDE_c(pred_pose, target_pose)))

    val_s_scores.append(avg_epoch_val_speed_loss.avg)
    print('| validation_speed_loss: %.2f' % avg_epoch_val_speed_loss.avg,
          '| validation_mask_loss: %.2f' % avg_epoch_val_pose_loss.avg,
          '| ade_val: %.2f' % ade_val.avg, '| fde_val: %.2f' % fde_val.avg,
          '| epoch_time.avg:%.2f' % (time.time() - start))
    sys.stdout.flush()


def regenerate_entire_pose(global_pose: torch.Tensor, local_pose: torch.Tensor):
    for i in len(local_pose):
        for j, pose in enumerate(local_pose[i]):
            for k in range(13):
                pose[2*k:2*(k+1)] = torch.add(global_pose[i][j], pose[2*k:2*(k+1)])
    return torch.cat((global_pose, local_pose), 2)


if __name__ == '__main__':
    opt = parse_option()
    _, val_loader = set_dataloader(opt)
    opt.model__name = 'de_global'
    global_model = set_model(opt)
    opt.model__name = 'de_local'
    local_model = set_model(opt)
    if opt.load_ckpt is not None:
        global_model = load_model(opt, global_model)
        local_model = load_model(opt, local_model)

    predict(val_loader, global_model, local_model)
