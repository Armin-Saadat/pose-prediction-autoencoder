import os
import sys
import time
import torch
import torch.nn as nn

from metrices import ADE_c, FDE_c, ADE_3d, FDE_3d
from utils import speed2pos, AverageMeter, speed2pos3d
from utils import set_dataloader, set_model, set_optimizer, set_scheduler, save_model, load_model
from option import parse_option


def train_global_disentangling(train_loader, val_loader, model, optimizer, scheduler, opt):
    training_start = time.time()
    l1e = nn.L1Loss()
    bce = nn.BCELoss()

    train_s_scores = []
    val_s_scores = []
    for epoch in range(1, opt.epochs + 1):
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
            obs_s = obs_s[:, :, :2].to(device=opt.device)
            target_s = target_s[:, :, :2].to(device=opt.device)
            obs_pose = obs_pose[:, :, :2].to(device=opt.device)
            target_pose = target_pose[:, :, :2].to(device=opt.device)
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

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_{name}_epoch_{epoch}.pth'.format(name=opt.name, epoch=epoch))
            save_model(model, optimizer, opt, idx, save_file)

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

        if (epoch + 1) % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_{name}_epoch_{epoch}.pth'.format(name=opt.name, epoch=epoch))
            save_model(model, optimizer, opt, idx, save_file)

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
    print('TRAINING 3dpw DONE in {} !'.format(time.time() - training_start))


if __name__ == '__main__':
    opt = parse_option()
    train_loader, val_loader = set_dataloader(opt)
    model = set_model(opt)
    if opt.load_ckpt is not None:
        model = load_model(opt, model)
    optimizer = set_optimizer(opt, model)
    scheduler = set_scheduler(opt, optimizer)
    if opt.dataset_name == 'posetrack':
        train_postrack(train_loader, val_loader, model, optimizer, scheduler, opt)
    else:
        train_3dpw(train_loader, val_loader, model, optimizer, scheduler, opt)
