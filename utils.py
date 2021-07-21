import torch
import torch.optim as optim

from models.lstm_vel_posetrack import LSTM_Vel_Posetrack
from models.lstm_vel_3dpw import LSTM_Vel_3dpw
from models.de_global_posetrack import DE_Global_Posetrack
from dataloader.lstm_vel_3dpw import data_loader
from dataloader.lstm_vel_posetrack import data_loader_posetrack


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_model(opt):
    model = set_model(opt)
    return load_model(opt, model)


def set_model(opt):
    if opt.model_name == 'lstm_vel':
        if opt.dataset_name == 'posetrack':
            return LSTM_Vel_Posetrack(opt).to(opt.device)
        else:
            return LSTM_Vel_3dpw(opt).to(opt.device)
    elif opt.model_name == 'disentangling':
        if opt.dataset_name == 'posetrack':
            return DE_Global_Posetrack(opt).to(opt.device)
        else:
            return None


def load_model(opt, model):
    ckpt = torch.load(opt.load_ckpt, map_location='cpu')
    state_dict = ckpt['model']
    if torch.cuda.is_available():
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def set_optimizer(opt, model):
    return optim.Adam(model.parameters(), lr=opt.learning_rate)


def set_scheduler(opt, optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=opt.lr_decay_rate, patience=10, threshold=1e-8,
                                                verbose=True)


def set_dataloader(opt):
    if opt.dataset_name == 'posetrack':
        train_loader = data_loader_posetrack(opt, "train", opt.dataset_name + "_")
        validation_loader = data_loader_posetrack(opt, "valid", opt.dataset_name + "_")
    else:
        train_loader = data_loader(opt, "train", opt.dataset_name + "_")
        validation_loader = data_loader(opt, 'valid', opt.dataset_name + "_")
    return train_loader, validation_loader


def speed2pos(preds, obs_p):
    pred_pos = torch.zeros(preds.shape[0], preds.shape[1], preds.shape[2]).to('cuda')
    current = obs_p[:, -1, :]

    for i in range(preds.shape[1]):
        pred_pos[:, i, :] = current + preds[:, i, :]
        current = pred_pos[:, i, :]

    for i in range(preds.shape[2]):
        pred_pos[:, :, i] = torch.min(pred_pos[:, :, i],
                                      1920 * torch.ones(pred_pos.shape[0], pred_pos.shape[1], device='cuda'))
        pred_pos[:, :, i] = torch.max(pred_pos[:, :, i],
                                      torch.zeros(pred_pos.shape[0], pred_pos.shape[1], device='cuda'))

    return pred_pos


def speed2pos3d(preds, obs_p):
    pred_pos = torch.zeros(preds.shape[0], preds.shape[1], preds.shape[2]).to('cuda')
    current = obs_p[:, -1, :]

    for i in range(preds.shape[1]):
        pred_pos[:, i, :] = current + preds[:, i, :]
        current = pred_pos[:, i, :]

    return pred_pos
