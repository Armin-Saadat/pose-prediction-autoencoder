import os
import sys
import time
import torch
import torch.nn as nn

from utils.metrices import ADE_c, FDE_c, mask_accuracy
from utils.others import speed2pos_local, AverageMeter
from utils.others import set_dataloader, set_model, set_optimizer, set_scheduler, save_model, load_model
from utils.option import parse_option


def train(train_loader, val_loader, model, optimizer, scheduler, opt):
    training_start = time.time()
    l1e = nn.L1Loss()
    bce = nn.BCELoss()
    train_s_scores = []
    val_s_scores = []
    for epoch in range(opt.epochs):
        start = time.time()
        avg_epoch_train_speed_loss = AverageMeter()
        avg_epoch_val_speed_loss = AverageMeter()
        avg_epoch_train_mask_loss = AverageMeter()
        avg_epoch_val_mask_loss = AverageMeter()
        avg_epoch_train_mask_acc = AverageMeter()
        avg_epoch_val_mask_acc = AverageMeter()
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
            batch_size = obs_s.shape[0]
            model.zero_grad()
            (speed_preds, mask_preds) = model(pose=obs_pose, vel=obs_s, mask=obs_mask)
            speed_loss = l1e(speed_preds, target_s)
            mask_loss = bce(mask_preds, target_mask)
            mask_acc = mask_accuracy(mask_preds, target_mask)

            preds_p = speed2pos_local(speed_preds, obs_pose)
            ade_train.update(val=float(ADE_c(preds_p, target_pose)), n=batch_size)
            fde_train.update(val=FDE_c(preds_p, target_pose), n=batch_size)

            loss = 1 * speed_loss + 0.3 * mask_loss
            loss.backward()
            optimizer.step()
            avg_epoch_train_speed_loss.update(val=float(speed_loss), n=batch_size)
            avg_epoch_train_mask_loss.update(val=float(mask_loss), n=batch_size)
            avg_epoch_train_mask_acc.update(val=float(mask_acc), n=batch_size)

        if (epoch + 1) % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, '{name}_epoch{epoch}.pth'.format(name=opt.name, epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
        train_s_scores.append(avg_epoch_train_speed_loss.avg)

        for idx, (obs_s, target_s, obs_pose, target_pose, obs_mask, target_mask) in enumerate(val_loader):
            obs_s = obs_s.to(device='cuda')
            target_s = target_s.to(device='cuda')
            obs_pose = obs_pose.to(device='cuda')
            target_pose = target_pose.to(device='cuda')
            obs_mask = obs_mask.to(device='cuda')
            target_mask = target_mask.to(device='cuda')
            batch_size = obs_s.shape[0]

            with torch.no_grad():
                (speed_preds, mask_preds) = model(pose=obs_pose, vel=obs_s, mask=obs_mask)
                speed_loss = l1e(speed_preds, target_s)
                mask_loss = bce(mask_preds, target_mask)
                mask_acc = mask_accuracy(mask_preds, target_mask)
                avg_epoch_val_speed_loss.update(val=float(speed_loss), n=batch_size)
                avg_epoch_val_mask_loss.update(val=float(mask_loss), n=batch_size)
                avg_epoch_val_mask_acc.update(val=float(mask_acc), n=batch_size)
                preds_p = speed2pos_local(speed_preds, obs_pose)
                ade_val.update(val=float(ADE_c(preds_p, target_pose)), n=batch_size)
                fde_val.update(val=float(FDE_c(preds_p, target_pose)), n=batch_size)

        val_s_scores.append(avg_epoch_val_speed_loss.avg)
        scheduler.step(avg_epoch_train_speed_loss.avg)
        print('e:', epoch, '| train_speed_loss: %.2f' % avg_epoch_train_speed_loss.avg,
              '| validation_speed_loss: %.2f' % avg_epoch_val_speed_loss.avg,
              '| train_mask_loss: %.2f' % avg_epoch_train_mask_loss.avg,
              '| validation_mask_loss: %.2f' % avg_epoch_val_mask_loss.avg,
              '| train_mask_acc: %.2f' % avg_epoch_train_mask_acc.avg,
              '| validation_mask_acc: %.2f' % avg_epoch_val_mask_acc.avg,
              '| ade_train: %.2f' % ade_train.avg,
              '| ade_val: %.2f' % ade_val.avg, '| fde_train: %.2f' % fde_train.avg, '| fde_val: %.2f' % fde_val.avg,
              '| epoch_time.avg:%.2f' % (time.time() - start))
        sys.stdout.flush()
    print("*" * 100)
    print('TRAINING Postrack DONE in:{}!'.format(time.time() - training_start))


if __name__ == '__main__':
    opt = parse_option('de_local', 'posetrack')
    train_loader, val_loader = set_dataloader(opt)
    model = set_model(opt)
    if opt.load_ckpt is not None:
        model = load_model(opt)
    optimizer = set_optimizer(opt, model)
    scheduler = set_scheduler(opt, optimizer)
    train(train_loader, val_loader, model, optimizer, scheduler, opt)
