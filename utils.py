import torch
import torch.optim as optim

from models.lstm_vel_posetrack import LSTMVelPosetrack
from models.lstm_vel_3dpw import LSTMVel3dpw
from models.de_global_posetrack import DEGlobalPosetrack
from models.de_local_posetrack import DELocalPosetrack
from dataloader.lstm_vel_dataloader import data_loader_lstm_vel
from dataloader.de_global_dataloader import data_loader_de_global
from dataloader.de_local_dataloader import data_loader_de_local
from dataloader.de_predict_dataloader import data_loader_de_predict


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
            return LSTMVelPosetrack(opt).to(opt.device)
        else:
            return LSTMVel3dpw(opt).to(opt.device)
    elif opt.model_name == 'de_global':
        if opt.dataset_name == 'posetrack':
            return DEGlobalPosetrack(opt).to(opt.device)
        else:
            return None
    elif opt.model_name == 'de_local':
        if opt.dataset_name == 'posetrack':
            return DELocalPosetrack(opt).to(opt.device)
        else:
            return None


def load_model(opt, model, load_ckpt=None):
    if load_ckpt:
        ckpt = torch.load(load_ckpt, map_location='cpu')
    else:
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
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=opt.lr_decay_rate, patience=40, threshold=1e-8,
                                                verbose=True)


def set_dataloader(opt):
    if opt.model_name == 'lstm_vel':
        train_loader = data_loader_lstm_vel(opt, "train", opt.dataset_name + "_")
        validation_loader = data_loader_lstm_vel(opt, "valid", opt.dataset_name + "_")
    elif opt.model_name == 'de_global':
        train_loader = data_loader_de_global(opt, "train", opt.dataset_name + "_")
        validation_loader = data_loader_de_global(opt, "valid", opt.dataset_name + "_")
    elif opt.model_name == 'de_local':
        train_loader = data_loader_de_local(opt, "train", opt.dataset_name + "_")
        validation_loader = data_loader_de_local(opt, "valid", opt.dataset_name + "_")
    elif opt.model_name == 'de_predict':
        train_loader = None
        validation_loader = data_loader_de_predict(opt, "valid", opt.dataset_name + "_")
    else:
        train_loader = None
        validation_loader = None
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


def speed2pos_local(preds, obs_p):
    pred_pos = torch.zeros(preds.shape[0], preds.shape[1], preds.shape[2]).to('cuda')
    current = obs_p[:, -1, :]

    for i in range(preds.shape[1]):
        pred_pos[:, i, :] = current + preds[:, i, :]
        current = pred_pos[:, i, :]

    return pred_pos


def speed2pos3d(preds, obs_p):
    pred_pos = torch.zeros(preds.shape[0], preds.shape[1], preds.shape[2]).to('cuda')
    current = obs_p[:, -1, :]

    for i in range(preds.shape[1]):
        pred_pos[:, i, :] = current + preds[:, i, :]
        current = pred_pos[:, i, :]

    return pred_pos
