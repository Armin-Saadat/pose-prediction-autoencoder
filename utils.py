from models.lstm_vel_posetrack import LSTM_Vel_Posetrack
from models.lstm_vel_3dpw import LSTM_Vel_3dpw
from models.de_global_posetrack import DE_Global_Posetrack
import torch
import torch.optim as optim

import pandas as pd
from ast import literal_eval



def set_loader(opt):
    if opt.dataset_name == 'posetrack':
        train_loader = data_loader_posetrack(opt, "train", opt.dataset_name + "_")
        validation_loader = data_loader_posetrack(opt, "valid", opt.dataset_name + "_")
    else:
        train_loader = data_loader(opt, "train", opt.dataset_name + "_")
        validation_loader = data_loader(opt, 'valid', opt.dataset_name + "_")

    return train_loader, validation_loader


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


def get_model(opt):
    model = set_model(opt)
    return load_model(opt, model)


def set_optimizer(opt, model):
    return optim.Adam(model.parameters(), lr=opt.learning_rate)


def set_scheduler(opt, optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=opt.lr_decay_rate, patience=10,
                                                threshold=1e-8, verbose=True)


class myDataset(torch.utils.data.Dataset):
    def __init__(self, args, dtype, fname):

        self.args = args
        self.dtype = dtype
        print("Loading", self.dtype)
        sequence_centric = pd.read_csv(fname + self.dtype + ".csv")
        df = sequence_centric.copy()
        for v in list(df.columns.values):
            print(v + ' loaded')
            try:
                df.loc[:, v] = df.loc[:, v].apply(lambda x: literal_eval(x))
            except:
                continue
        sequence_centric[df.columns] = df[df.columns]
        self.data = sequence_centric.copy().reset_index(drop=True)

        print('*' * 30)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        seq = self.data.iloc[index]
        outputs = []

        obs = torch.tensor([seq.Pose[i] for i in range(0, self.args.input, self.args.skip)])
        obs_speed = (obs[1:] - obs[:-1])
        outputs.append(obs_speed)
        true = torch.tensor([seq.Future_Pose[i] for i in range(0, self.args.output, self.args.skip)])
        true_speed = torch.cat(((true[0] - obs[-1]).unsqueeze(0), true[1:] - true[:-1]))
        outputs.append(true_speed)
        outputs.append(obs)
        outputs.append(true)

        return tuple(outputs)


def data_loader(args, data, fname):
    dataset = myDataset(args, data, fname)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory)

    return dataloader


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


class myDataset_posetrack(torch.utils.data.Dataset):
    def __init__(self, args, dtype, fname):
        self.args = args
        self.dtype = dtype
        self.fname = fname
        print("Loading", self.dtype)
        sequence_centric = pd.read_csv('processed_csvs/' + self.fname + self.dtype + ".csv")
        df = sequence_centric.copy()
        for v in list(df.columns.values):
            print(v + ' loaded')
            try:
                df.loc[:, v] = df.loc[:, v].apply(lambda x: literal_eval(x))
            except:
                continue
        sequence_centric[df.columns] = df[df.columns]
        self.data = sequence_centric.copy().reset_index(drop=True)
        print('*' * 30)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data.iloc[index]
        outputs = []
        obs = torch.tensor([seq.Pose[i] for i in range(0, self.args.input, self.args.skip)])
        obs_speed = (obs[1:] - obs[:-1])
        outputs.append(obs_speed)
        true = torch.tensor([seq.Future_Pose[i] for i in range(0, self.args.output, self.args.skip)])
        true_speed = torch.cat(((true[0] - obs[-1]).unsqueeze(0), true[1:] - true[:-1]))
        outputs.append(true_speed)
        outputs.append(obs)
        outputs.append(true)

        if self.fname == "posetrack_":
            obs_mask = torch.tensor([seq.Mask[i] for i in range(0, self.args.output, self.args.skip)])
            true_mask = torch.tensor([seq.Future_Mask[i] for i in range(0, self.args.output, self.args.skip)])
            outputs.append(obs_mask)
            outputs.append(true_mask)

        return tuple(outputs)


def data_loader_posetrack(args, data, fname):
    dataset = myDataset_posetrack(args, data, fname)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory)

    return dataloader







class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

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
