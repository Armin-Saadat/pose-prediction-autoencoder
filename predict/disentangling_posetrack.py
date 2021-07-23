import argparse
import torch
import torch.nn as nn
from utils.others import set_dataloader, set_model, load_model, AverageMeter, speed2pos, speed2pos_local
from utils.metrices import ADE_c, FDE_c
import time
import sys


def parse_option():
    parser = argparse.ArgumentParser('argument for predictions')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input', type=int, default=16)
    parser.add_argument('--output', type=int, default=14)
    parser.add_argument('--hidden_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hardtanh_limit', type=int, default=100)
    parser.add_argument('--load_local_ckpt', type=str)
    parser.add_argument('--load_global_ckpt', type=str)

    opt = parser.parse_args()
    opt.stride = opt.input
    opt.skip = 1
    opt.dataset_name = 'posetrack'
    opt.loader_shuffle = True
    opt.pin_memory = False
    return opt


def predict(loader, global_model, local_model):
    l1e = nn.L1Loss()
    bce = nn.BCELoss()
    val_s_scores = []

    start = time.time()
    avg_epoch_val_speed_loss = AverageMeter()
    avg_epoch_val_mask_loss = AverageMeter()

    ade_val = AverageMeter()
    fde_val = AverageMeter()
    for idx, (obs_velocities, target_velocities, obs_pose, target_pose, obs_mask, target_mask) in enumerate(loader):
        global_vel_obs = obs_velocities[:, :, :2].to(device='cuda')
        local_vel_obs = obs_velocities[:, :, 2:].to(device='cuda')
        global_vel_targets = target_velocities[:, :, :2].to(device='cuda')
        local_vel_targets = target_velocities[:, :, 2:].to(device='cuda')
        global_pose_obs = obs_pose[:, :, :2].to(device='cuda')
        local_pose_obs = obs_pose[:, :, 2:].to(device='cuda')
        target_pose = target_pose.to(device='cuda')
        mask_obs = obs_mask.to(device='cuda')
        mask_target = target_mask.to(device='cuda')
        with torch.no_grad():
            global_vel_preds = global_model(pose=global_pose_obs, vel=global_vel_obs)
            (local_vel_preds, mask_preds) = local_model(pose=local_pose_obs, vel=local_vel_obs, mask=mask_obs)
            local_speed_loss = l1e(local_vel_preds, local_vel_targets)
            global_speed_loss = l1e(global_vel_preds, global_vel_targets)
            mask_loss = bce(mask_preds, mask_target)
            avg_epoch_val_mask_loss.update(val=float(mask_loss))
            avg_epoch_val_speed_loss.update(val=float((global_speed_loss + 13 * local_speed_loss) / 14))

            global_pose_pred = speed2pos(global_vel_preds, global_pose_obs)
            local_pose_pred = speed2pos_local(local_vel_preds, local_pose_obs)
            # now we have to make a prediction
            pose_pred = regenerate_entire_pose(global_pose_pred, local_pose_pred)
            ade_val.update(val=float(ADE_c(pose_pred, target_pose)))
            fde_val.update(val=FDE_c(pose_pred, target_pose))

    val_s_scores.append(avg_epoch_val_speed_loss.avg)
    print('| validation_speed_loss: %.2f' % avg_epoch_val_speed_loss.avg,
          '| validation_mask_loss: %.2f' % avg_epoch_val_mask_loss.avg,
          '| ade_val: %.2f' % ade_val.avg, '| fde_val: %.2f' % fde_val.avg,
          '| epoch_time.avg:%.2f' % (time.time() - start))
    sys.stdout.flush()


def regenerate_entire_pose(global_pose: torch.Tensor, local_pose: torch.Tensor):
    for i in range(len(local_pose)):  # iterate over batch size
        for j, pose in enumerate(local_pose[i]):  # iterate over frames
            for k in range(13):
                pose[2 * k:2 * (k + 1)] = torch.add(pose[2 * k:2 * (k + 1)], global_pose[i][j], )
    return torch.cat((global_pose, local_pose), 2)


if __name__ == '__main__':
    opt = parse_option()
    opt.model_name = 'de_predict'
    _, val_loader = set_dataloader(opt)
    opt.model_name = 'de_global'
    global_model = set_model(opt)
    opt.model_name = 'de_local'
    local_model = set_model(opt)
    if opt.load_local_ckpt is not None:
        global_model = load_model(opt, global_model, opt.load_global_ckpt)
        local_model = load_model(opt, local_model, opt.load_local_ckpt)
    else:
        raise EnvironmentError
    predict(val_loader, global_model, local_model)
