import argparse
import torch
import torch.nn as nn
from utils.others import set_dataloader, set_model, load_model, AverageMeter, speed2pos
from utils.metrices import ADE_c, FDE_c, mask_accuracy
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
    parser.add_argument('--load_lstm_vel_ckpt', type=str)
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


def predict(loader, global_model, lstm_vel_model):
    lstm_vel_model.eval()
    global_model.eval()
    l1e = nn.L1Loss()
    bce = nn.BCELoss()
    start = time.time()
    avg_epoch_speed_loss = AverageMeter()
    avg_epoch_mask_loss = AverageMeter()
    avg_epoch_mask_acc = AverageMeter()
    ade_val = AverageMeter()
    fde_val = AverageMeter()
    for idx, (obs_s, target_s, obs_pose, target_pose, obs_mask, target_mask) in enumerate(loader):
        global_vel_obs = obs_s[:, :, :2].to(device='cuda')
        global_pose_obs = obs_pose[:, :, :2].to(device='cuda')
        obs_s = obs_s.to(device='cuda')
        target_s = target_s.to(device='cuda')
        obs_pose = obs_pose.to(device='cuda')
        target_pose = target_pose.to(device='cuda')
        obs_mask = obs_mask.to(device='cuda')
        target_mask = target_mask.to(device='cuda')

        with torch.no_grad():
            speed_preds, _ = lstm_vel_model(pose=obs_pose, vel=obs_s, mask=obs_mask)
            global_vel_preds = global_model(pose=global_pose_obs, vel=global_vel_obs)
            m = obs_mask[:, -1:, :]
            mask_preds = torch.cat((m, m, m, m, m, m, m, m, m, m, m, m, m, m), 1)
            mask_loss = bce(mask_preds, target_mask)
            mask_acc = mask_accuracy(mask_preds, target_mask)

            speed_loss = l1e(speed_preds, target_s)

            avg_epoch_speed_loss.update(val=float(speed_loss), n=target_s.shape[0])
            avg_epoch_mask_loss.update(val=float(mask_loss), n=target_mask.shape[0])
            avg_epoch_mask_acc.update(val=float(mask_acc), n=target_mask.shape[0])
            global_pose_pred = speed2pos(global_vel_preds, global_pose_obs)
            pose_pred = speed2pos(speed_preds, obs_pose)
            preds_p = torch.cat( (global_pose_pred, pose_pred[:, :, 2:]), 2)
            ade_val.update(val=float(ADE_c(preds_p, target_pose)), n=target_pose.shape[0])
            fde_val.update(val=float(FDE_c(preds_p, target_pose)), n=target_pose.shape[0])

    print('| speed_loss: %.2f' % avg_epoch_speed_loss.avg,
          '| mask_loss: %.2f' % avg_epoch_mask_loss.avg,
          '| mask_acc: %.2f' % avg_epoch_mask_acc.avg,
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
    opt.model_name = 'lstm_vel'
    _, val_loader = set_dataloader(opt)
    lstm_vel_model = set_model(opt)
    opt.model_name = 'de_global'
    global_model = set_model(opt)
    if opt.load_global_ckpt is not None:
        global_model = load_model(opt, global_model, opt.load_global_ckpt)
    else:
        raise EnvironmentError
    if opt.load_lstm_vel_ckpt is not None:
        lstm_vel_model = load_model(opt, lstm_vel_model, opt.load_lstm_vel_ckpt)
    predict(val_loader, global_model, lstm_vel_model)
