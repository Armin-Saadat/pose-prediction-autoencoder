import time
import torch
import torch.nn as nn

from utils.metrices import ADE_3d, FDE_3d
from utils.others import AverageMeter, speed2pos3d
from utils.others import set_dataloader, set_model, set_optimizer, set_scheduler, load_model
from utils.option import parse_option


def train(train_loader, val_loader, model, optimizer, scheduler, opt):
    training_start = time.time()
    l1e = nn.L1Loss()
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
            batch_size = obs_s.shape[0]
            model.zero_grad()

            speed_preds = model(pose=obs_pose, vel=obs_s)
            speed_loss = l1e(speed_preds, target_s)

            preds_p = speed2pos3d(speed_preds, obs_pose)
            ade_train.update(val=float(ADE_3d(preds_p, target_pose)), n=batch_size)
            fde_train.update(val=float(FDE_3d(preds_p, target_pose)), n=batch_size)

            speed_loss.backward()
            optimizer.step()
            avg_epoch_train_speed_loss.update(float(speed_loss), n=batch_size)

        train_s_scores.append(avg_epoch_train_speed_loss.avg)
        counter = 0

        for idx, (obs_s, target_s, obs_pose, target_pose) in enumerate(val_loader):
            counter += 1
            obs_s = obs_s.to(device='cuda')
            target_s = target_s.to(device='cuda')
            obs_pose = obs_pose.to(device='cuda')
            target_pose = target_pose.to(device='cuda')
            batch_size = obs_s.shape[0]

            with torch.no_grad():
                (speed_preds,) = model(pose=obs_pose, vel=obs_s)

                speed_loss = l1e(speed_preds, target_s)
                avg_epoch_val_speed_loss.update(float(speed_loss), n=batch_size)

                preds_p = speed2pos3d(speed_preds, obs_pose)
                ade_train.update(float(ADE_3d(preds_p, target_pose)), n=batch_size)
                fde_val.update(FDE_3d(preds_p, target_pose), n=batch_size)

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
    opt = parse_option('lstm_vel', '3dpw')
    train_loader, val_loader = set_dataloader(opt)
    model = set_model(opt)
    if opt.load_ckpt is not None:
        model = load_model(opt)
    optimizer = set_optimizer(opt, model)
    scheduler = set_scheduler(opt, optimizer)
    train(train_loader, val_loader, model, optimizer, scheduler, opt)
