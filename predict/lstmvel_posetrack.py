import argparse
import json
import sys
import time

import torch
import torch.nn as nn

from utils.metrices import ADE_c, FDE_c, mask_accuracy
from utils.others import set_dataloader, load_model, AverageMeter, speed2pos


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
    parser.add_argument('--test_output', type=bool, default=False)

    opt = parser.parse_args()
    opt.stride = opt.input
    opt.skip = 1
    opt.dataset_name = 'posetrack'
    opt.loader_shuffle = True
    opt.pin_memory = False
    opt.model_name = 'lstm_vel'
    opt.device = 'cuda'

    return opt


def predict(loader, model, opt):
    model.eval()
    l1e = nn.L1Loss()
    bce = nn.BCELoss()
    start = time.time()
    avg_epoch_speed_loss = AverageMeter()
    avg_epoch_mask_loss = AverageMeter()
    avg_epoch_mask_acc = AverageMeter()
    ade_val = AverageMeter()
    fde_val = AverageMeter()
    for idx, (obs_s, target_s, obs_pose, target_pose, obs_mask, target_mask) in enumerate(loader):
        obs_s = obs_s.to(opt.device)
        target_s = target_s.to(opt.device)
        obs_pose = obs_pose.to(opt.device)
        target_pose = target_pose.to(opt.device)
        obs_mask = obs_mask.to(opt.device)
        target_mask = target_mask.to(opt.device)
        with torch.no_grad():
            speed_preds, _ = model(pose=obs_pose, vel=obs_s, mask=obs_mask)
            m = obs_mask[:, -1:, :]
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
    if opt.test_output:
        with open("../outputs/Posetrack/posetrack_test_in.json", "r") as read_file:
            data = json.load(read_file)

        with open("../outputs/Posetrack/posetrack_test_masks_in.json", "r") as read_file:
            data_m = json.load(read_file)

        out_data = []
        out_mask = []
        for i in range(len(data)):
            lp = []
            lm = []
            for j in range(len(data[i])):
                pose = torch.tensor(data[i][j]).unsqueeze(0).to(opt.device)
                mask = torch.tensor(data_m[i][j]).unsqueeze(0).to(opt.device)
                vel = pose[:, 1:] - pose[:, :-1]
                (speed_preds, _) = model(pose=pose, vel=vel, mask=mask)
                m = mask[:, -1:, :]
                mask_preds = torch.cat((m, m, m, m, m, m, m, m, m, m, m, m, m, m), 1)

                preds_p = speed2pos(speed_preds, pose)
                pred = preds_p.squeeze(0)
                mask_pred = mask_preds.squeeze(0)
                lp.append(pred.tolist())
                lm.append(mask_pred.detach().cpu().numpy().round().tolist())
            out_data.append(lp)
            out_mask.append(lm)
        with open('../outputs/Posetrack/posetrack_predictions_{}.json'.format(
                opt.load_ckpt.split('snapshots/')[1].split('.pth')[0]), 'w') as f:
            json.dump(out_data, f)
        with open('../outputs/Posetrack/posetrack_masks_{}.json'.format(opt.load_ckpt.split('snapshots/')[1].split('.pth')[0]),
                  'w') as f:
            json.dump(out_mask, f)


if __name__ == '__main__':
    opt = parse_option()
    load_ckpt = opt.load_ckpt
    _, val_loader = set_dataloader(opt)
    if opt.load_ckpt is not None:
        model = load_model(opt)
    else:
        raise EnvironmentError
    opt.load_ckpt = load_ckpt
    predict(val_loader, model, opt)
