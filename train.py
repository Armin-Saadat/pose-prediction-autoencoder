import torch
import torch.nn as nn

import sys
import argparse
import time
from utils import ADE_c, FDE_c, speed2pos, AverageMeter, ADE_3d, FDE_3d, speed2pos3d
from utils import set_loader, set_model, set_optimizer, set_scheduler


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=80,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.25,
                        help='decay rate for learning rate')
    parser.add_argument('--loader_shuffle', type=bool, default=False)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input_size', type=int, default=14)
    parser.add_argument('--output_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=1000)
    parser.add_argument('--hardtanh_limit', type=int, default=100)
    parser.add_argument('--dataset_name', type=str, default='posetrack', choices=['posetrack', '3dpw'])
    parser.add_argument('--model_name', type=str, default='lstm_vel', choices=['lstm_vel', 'disentangling'])
    parser.add_argument('--input', type=int, default=16)
    parser.add_argument('--output', type=int, default=14)

    opt = parser.parse_args()
    opt.stride = opt.input
    opt.skip = 1
    return opt


def train_postrack(train_loader, val_loader, model, optimizer, scheduler, opt):
    training_start = time.time()
    l1e = nn.L1Loss()
    bce = nn.BCELoss()

    train_s_scores = []
    val_s_scores = []
    for epoch in range(opt.epochs):
        start = time.time()
        avg_epoch_train_speed_loss = AverageMeter()
        avg_epoch_val_speed_loss = AverageMeter()
        avg_epoch_train_pose_loss = AverageMeter()
        avg_epoch_val_pose_loss = AverageMeter()

        ade_train = AverageMeter()
        ade_val = AverageMeter()
        fde_train = AverageMeter()
        fde_val = AverageMeter()

        for idx, (obs_s, target_s, obs_pose, target_pose, obs_mask, target_mask) in enumerate(train_loader):
            obs_s = obs_s.to(device=opt.device)
            target_s = target_s.to(device=opt.device)
            obs_pose = obs_pose.to(device=opt.device)
            target_pose = target_pose.to(device=opt.device)
            obs_mask = obs_mask.to(device=opt.device)
            target_mask = target_mask.to(device=opt.device)

            model.zero_grad()
            (speed_preds, mask_preds) = model(pose=obs_pose, vel=obs_s, mask=obs_mask)
            speed_loss = l1e(speed_preds, target_s)
            mask_loss = bce(mask_preds, target_mask)

            preds_p = speed2pos(speed_preds, obs_pose)
            ade_train.update(val=float(ADE_c(preds_p, target_pose)))
            fde_train.update(val=FDE_c(preds_p, target_pose))

            loss = 0.8 * speed_loss + 0.2 * mask_loss
            loss.backward()

            optimizer.step()
            avg_epoch_train_speed_loss.update(val=float(speed_loss))
            avg_epoch_train_pose_loss.update(val=float(mask_loss))

        train_s_scores.append(avg_epoch_train_speed_loss.avg)

        for idx, (obs_s, target_s, obs_pose, target_pose, obs_mask, target_mask) in enumerate(val_loader):
            obs_s = obs_s.to(device='cuda')
            target_s = target_s.to(device='cuda')
            obs_pose = obs_pose.to(device='cuda')
            target_pose = target_pose.to(device='cuda')
            obs_mask = obs_mask.to(device='cuda')
            target_mask = target_mask.to(device='cuda')

            with torch.no_grad():
                (speed_preds, mask_preds) = model(pose=obs_pose, vel=obs_s, mask=obs_mask)

                speed_loss = l1e(speed_preds, target_s)
                mask_loss = bce(mask_preds, target_mask)

                avg_epoch_val_speed_loss.update(val=float(speed_loss))
                avg_epoch_val_pose_loss.update(val=float(mask_loss))

                preds_p = speed2pos(speed_preds, obs_pose)
                ade_val.update(val=float(ADE_c(preds_p, target_pose)))
                fde_val.update(val=float(FDE_c(preds_p, target_pose)))

        val_s_scores.append(avg_epoch_val_speed_loss.avg)

        scheduler.step(avg_epoch_val_speed_loss.avg)

        print('e:', epoch, '| train_speed_loss: %.2f' % avg_epoch_train_speed_loss.avg,
              '| validation_speed_loss: %.2f' % avg_epoch_val_speed_loss.avg,
              '| train_mask_loss: %.2f' % avg_epoch_train_pose_loss.avg,
              '| validation_mask_loss: %.2f' % avg_epoch_val_pose_loss.avg, '| ade_train: %.2f' % ade_train.avg,
              '| ade_val: %.2f' % ade_val.avg, '| fde_train: %.2f' % fde_train.avg, '| fde_val: %.2f' % fde_val.avg,
              '| epoch_time.avg:%.2f' % (time.time() - start))
        sys.stdout.flush()

    print("*" * 100)
    print('TRAINING Postrack DONE in:{}!'.format(time.time() - training_start))


def train_3dpw(train_loader, val_loader, model, optimizer, scheduler, opt):
    training_start = time.time()

    l1e = nn.L1Loss()
    bce = nn.BCELoss()

    train_s_scores = []
    val_s_scores = []
    for epoch in range(opt.epochs):
        start = time.time()

        avg_epoch_train_speed_loss = AverageMeter()
        avg_epoch_val_speed_loss = AverageMeter()

        ade_train = AverageMeter()
        ade_val = AverageMeter()
        fde_train = AverageMeter()
        fde_val = AverageMeter()

        counter = 0

        for idx, (obs_s, target_s, obs_pose, target_pose) in enumerate(train_loader):
            counter += 1

            obs_s = obs_s.to(device='cuda')
            target_s = target_s.to(device='cuda')
            obs_pose = obs_pose.to(device='cuda')
            target_pose = target_pose.to(device='cuda')

            model.zero_grad()

            (speed_preds,) = model(pose=obs_pose, vel=obs_s)

            speed_loss = l1e(speed_preds, target_s)

            preds_p = speed2pos3d(speed_preds, obs_pose)
            ade_train.update(val=float(ADE_3d(preds_p, target_pose)))
            fde_train.update(val=float(FDE_3d(preds_p, target_pose)))

            speed_loss.backward()

            optimizer.step()

            avg_epoch_train_speed_loss.update(float(speed_loss))

        train_s_scores.append(avg_epoch_train_speed_loss.avg)

        counter = 0

        for idx, (obs_s, target_s, obs_pose, target_pose) in enumerate(val_loader):
            counter += 1
            obs_s = obs_s.to(device='cuda')
            target_s = target_s.to(device='cuda')
            obs_pose = obs_pose.to(device='cuda')
            target_pose = target_pose.to(device='cuda')

            with torch.no_grad():
                (speed_preds,) = model(pose=obs_pose, vel=obs_s)

                speed_loss = l1e(speed_preds, target_s)
                avg_epoch_val_speed_loss.update(float(speed_loss))

                preds_p = speed2pos3d(speed_preds, obs_pose)
                ade_train.update(float(ADE_3d(preds_p, target_pose)))
                fde_val.update(FDE_3d(preds_p, target_pose))

        val_s_scores.append(avg_epoch_val_speed_loss.avg)

        scheduler.step(avg_epoch_train_speed_loss.avg)

        print('e:', epoch, '| train_speed_loss: %.2f' % avg_epoch_train_speed_loss.avg,
              '| validation_speed_loss: %.2f' % avg_epoch_val_speed_loss.avg,
              '| ade_train: %.2f' % ade_train.avg,
              '| ade_val: %.2f' % ade_val.avg, '| fde_train: %.2f' % fde_train.avg, '| fde_val: %.2f' % fde_val.avg,
              '| epoch_time.avg:%.2f' % (time.time() - start))

    print('*' * 100)
    # print('Saving ...')
    # torch.save(net.state_dict(), args.model_path)
    print('TRAINING 3dpw DONE in {} !'.format(time.time() - training_start))


if __name__ == '__main__':
    opt = parse_option()
    train_loader, val_loader = set_loader(opt)
    model = set_model(opt)
    optimizer = set_optimizer(opt, model)
    scheduler = set_scheduler(opt, optimizer)
    if opt.dataset_name == 'posetrack':
        train_postrack(train_loader, val_loader, model, optimizer, scheduler, opt)
    else:
        train_3dpw(train_loader, val_loader, model, optimizer, scheduler, opt)
