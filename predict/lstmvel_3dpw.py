import argparse
import torch
import torch.nn as nn
import time
from utils.others import set_dataloader, set_model, load_model, AverageMeter, speed2pos3d
from utils.metrices import ADE_3d, FDE_3d
import sys


def parse_option():
    parser = argparse.ArgumentParser('argument for predictions')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input', type=int, default=16)
    parser.add_argument('--output', type=int, default=14)
    parser.add_argument('--hidden_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hardtanh_limit', type=int, default=100)
    parser.add_argument('--load_ckpt', type=str)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dropout_encoder', type=float, default=0)
    parser.add_argument('--dropout_pose_decoder', type=float, default=0)
    parser.add_argument('--dropout_mask_decoder', type=float, default=0)
    opt = parser.parse_args()
    opt.stride = opt.input
    opt.skip = 1
    opt.dataset_name = '3dpw'
    opt.loader_shuffle = True
    opt.pin_memory = False
    opt.model_name = 'lstm_vel'
    return opt


def predict(loader, model):
    l1e = nn.L1Loss()
    start = time.time()
    avg_epoch_speed_loss = AverageMeter()
    ade_val = AverageMeter()
    fde_val = AverageMeter()
    for idx, (obs_s, target_s, obs_pose, target_pose) in loader:
        obs_s = obs_s.to(device='cuda')
        target_s = target_s.to(device='cuda')
        obs_pose = obs_pose.to(device='cuda')
        target_pose = target_pose.to(device='cuda')
        with torch.no_grad():
            speed_preds = model(pose=obs_pose, vel=obs_s)
            speed_loss = l1e(speed_preds, target_s)
            avg_epoch_speed_loss.update(val=float(speed_loss), n=target_s.shape[0])

            preds_p = speed2pos3d(speed_preds, obs_pose)
            ade_val.update(val=float(ADE_3d(preds_p, target_pose)), n=target_pose.shape[0])
            fde_val.update(val=float(FDE_3d(preds_p, target_pose)), n=target_pose.shape[0])

    print('| speed_loss: %.2f' % avg_epoch_speed_loss.avg,
          '| ade_val: %.2f' % ade_val.avg, '| fde_val: %.2f' % fde_val.avg,
          '| epoch_time.avg:%.2f' % (time.time() - start))
    sys.stdout.flush()


if __name__ == '__main__':
    opt = parse_option()
    _, val_loader = set_dataloader(opt)
    model = set_model(opt)
    if opt.load_ckpt is not None:
        model = load_model(opt, model)

    predict(val_loader, model)
