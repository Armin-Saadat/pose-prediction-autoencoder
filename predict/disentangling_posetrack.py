import json
import argparse
import torch
import torch.nn as nn
from utils.others import set_dataloader, set_model, load_model, AverageMeter, speed2pos, speed2pos_local
from utils.metrices import ADE_c, FDE_c, mask_accuracy
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
    opt.device = 'cuda'
    return opt


def predict(loader, global_model, local_model, opt):
    global_model.eval()
    local_model.eval()
    l1e = nn.L1Loss()
    bce = nn.BCELoss()
    start = time.time()
    avg_epoch_speed_loss = AverageMeter()
    avg_epoch_mask_loss = AverageMeter()
    avg_epoch_mask_acc = AverageMeter()
    ade_val = AverageMeter()
    fde_val = AverageMeter()
    for idx, (obs_velocities, target_velocities, obs_pose, target_pose, obs_mask, target_mask) in enumerate(loader):
        global_vel_obs = obs_velocities[:, :, :2].to(opt.device)
        local_vel_obs = obs_velocities[:, :, 2:].to(opt.device)
        global_vel_targets = target_velocities[:, :, :2].to(opt.device)
        local_vel_targets = target_velocities[:, :, 2:].to(opt.device)
        global_pose_obs = obs_pose[:, :, :2].to(opt.device)
        local_pose_obs = obs_pose[:, :, 2:].to(opt.device)
        target_pose = target_pose.to(opt.device)
        obs_mask = obs_mask.to(opt.device)
        target_mask = target_mask.to(opt.device)
        with torch.no_grad():
            global_vel_preds = global_model(pose=global_pose_obs, vel=global_vel_obs)
            local_vel_preds, _ = local_model(pose=local_pose_obs, vel=local_vel_obs, mask=obs_mask)
            m = obs_mask[:, -1:, :]
            mask_preds = torch.cat((m, m, m, m, m, m, m, m, m, m, m, m, m, m), 1)
            local_speed_loss = l1e(local_vel_preds, local_vel_targets)
            global_speed_loss = l1e(global_vel_preds, global_vel_targets)
            avg_epoch_speed_loss.update(val=float((global_speed_loss + 13 * local_speed_loss) / 14),
                                        n=global_vel_targets.shape[0])

            mask_loss = bce(mask_preds, target_mask)
            avg_epoch_mask_loss.update(val=float(mask_loss), n=target_mask.shape[0])

            mask_acc = mask_accuracy(mask_preds, target_mask)
            avg_epoch_mask_acc.update(val=float(mask_acc), n=target_mask.shape[0])

            global_pose_pred = speed2pos(global_vel_preds, global_pose_obs)
            local_pose_pred = speed2pos_local(local_vel_preds, local_pose_obs)

            pose_pred = regenerate_entire_pose(global_pose_pred, local_pose_pred)
            ade_val.update(val=float(ADE_c(pose_pred, target_pose)), n=target_pose.shape[0])
            fde_val.update(val=FDE_c(pose_pred, target_pose), n=target_pose.shape[0])

    print('| speed_loss: %.2f' % avg_epoch_speed_loss.avg,
          '| mask_loss: %.2f' % avg_epoch_mask_loss.avg,
          '| mask_acc: %.2f' % avg_epoch_mask_acc.avg,
          '| ade_val: %.2f' % ade_val.avg, '| fde_val: %.2f' % fde_val.avg,
          '| epoch_time.avg:%.2f' % (time.time() - start))
    sys.stdout.flush()
    if opt.test_output:
        with open("./Posetrack/posetrack_test_in.json", "r") as read_file:
            data = json.load(read_file)

        with open("./Posetrack/posetrack_test_masks_in.json", "r") as read_file:
            data_m = json.load(read_file)

        out_data = []
        out_mask = []
        for i in range(len(data)):
            lp = []
            lm = []
            for j in range(len(data[i])):
                # TODO: bug below
                pose = torch.tensor(data[i][j]).unsqueeze(0).to(opt.device)
                mask = torch.tensor(data_m[i][j]).unsqueeze(0).to(opt.device)
                vel = pose[:, 1:] - pose[:, :-1]
                global_vel_preds_t = global_model(pose=global_pose_obs, vel=global_vel_obs)
                local_vel_preds_t, _ = local_model(pose=local_pose_obs, vel=local_vel_obs, mask=mask)
                global_pose_pred_t = speed2pos(global_vel_preds_t, global_pose_obs)
                local_pose_pred_t = speed2pos_local(local_vel_preds_t, local_pose_obs)
                pose_pred_t = regenerate_entire_pose(global_pose_pred_t, local_pose_pred_t)
                m = mask[:, -1:, :]
                mask_preds_test = torch.cat((m, m, m, m, m, m, m, m, m, m, m, m, m, m), 1)

                pred = pose_pred_t.squeeze(0)
                mask_pred = mask_preds_test.squeeze(0)
                lp.append(pred.tolist())
                lm.append(mask_pred.detach().cpu().numpy().round().tolist())
            out_data.append(lp)
            out_mask.append(lm)
        with open('../outputs/Posetrack/posetrack_predictions_{}.json'.format('disentangling'), 'w') as f:
            json.dump(out_data, f)
        with open('../outputs/posetrack_masks_{}.json'.format('disentangling'), 'w') as f:
            json.dump(out_mask, f)


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
    if opt.load_global_ckpt is not None:
        global_model = load_model(opt, opt.load_global_ckpt)
    else:
        raise EnvironmentError
    opt.model_name = 'de_local'
    if opt.load_local_ckpt is not None:
        local_model = load_model(opt, opt.load_local_ckpt)
    else:
        raise EnvironmentError
    predict(val_loader, global_model, local_model, opt)
