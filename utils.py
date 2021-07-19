from models.lstm_posetrack import LSTM_posetrack
from models.lstm_3dpw import LSTM_vel3d
import torch
import torch.optim as optim
import matplotlib
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from ast import literal_eval
import cv2


def set_loader(opt):
    if opt.dataset_name == 'posetrack':
        train_loader = data_loader_posetrack(opt, "train", opt.dataset_name + "_")
        validation_loader = data_loader_posetrack(opt, "valid", opt.dataset_name + "_")
    else:
        train_loader = data_loader(opt, "train", opt.dataset_name + "_")
        validation_loader = data_loader(opt, 'valid', opt.dataset_name + "_")

    return train_loader, validation_loader


def set_model(opt):
    if opt.dataset_name == 'posetrack':
        return LSTM_posetrack(opt).to(opt.device)
    else:
        return LSTM_vel3d(opt).to(opt.device)


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


def ADE_c(pred, true):
    b, n, p = pred.size()[0], pred.size()[1], pred.size()[2]
    #     print(b,n,p)
    pred = torch.reshape(pred, (b, n, int(p / 2), 2))
    true = torch.reshape(true, (b, n, int(p / 2), 2))

    displacement = torch.sqrt((pred[:, :, :, 0] - true[:, :, :, 0]) ** 2 + (pred[:, :, :, 1] - true[:, :, :, 1]) ** 2)
    ade = torch.mean(torch.mean(displacement, dim=1))

    return ade


def FDE_c(pred, true):
    b, n, p = pred.size()[0], pred.size()[1], pred.size()[2]
    #     print(b,n,p)
    pred = torch.reshape(pred, (b, n, int(p / 2), 2))
    true = torch.reshape(true, (b, n, int(p / 2), 2))

    displacement = torch.sqrt(
        (pred[:, -1, :, 0] - true[:, -1, :, 0]) ** 2 + (pred[:, -1, :, 1] - true[:, -1, :, 1]) ** 2)

    fde = torch.mean(torch.mean(displacement, dim=1))

    return fde


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


def ADE_3d(pred, true):
    b, n, p = pred.size()[0], pred.size()[1], pred.size()[2]
    #     print(b,n,p)
    pred = torch.reshape(pred, (b, n, int(p / 3), 3))
    true = torch.reshape(true, (b, n, int(p / 3), 3))

    displacement = torch.sqrt((pred[:, :, :, 0] - true[:, :, :, 0]) ** 2 + (pred[:, :, :, 1] - true[:, :, :, 1]) ** 2)
    ade = torch.mean(torch.mean(displacement, dim=1))

    return ade


def FDE_3d(pred, true):
    b, n, p = pred.size()[0], pred.size()[1], pred.size()[2]
    #     print(b,n,p)
    pred = torch.reshape(pred, (b, n, int(p / 3), 3))
    true = torch.reshape(true, (b, n, int(p / 3), 3))

    displacement = torch.sqrt(
        (pred[:, -1, :, 0] - true[:, -1, :, 0]) ** 2 + (pred[:, -1, :, 1] - true[:, -1, :, 1]) ** 2)

    fde = torch.mean(torch.mean(displacement, dim=1))

    return fde


def speed2pos3d(preds, obs_p):
    pred_pos = torch.zeros(preds.shape[0], preds.shape[1], preds.shape[2]).to('cuda')
    current = obs_p[:, -1, :]

    for i in range(preds.shape[1]):
        pred_pos[:, i, :] = current + preds[:, i, :]
        current = pred_pos[:, i, :]

    return pred_pos


def VIM(GT, pred, dataset_name, mask):
    """
    Visibilty Ignored Metric
    Inputs:
        GT: Ground truth data - array of shape (pred_len, #joint*(2D/3D))
        pred: Prediction data - array of shape (pred_len, #joint*(2D/3D))
        dataset_name: Dataset name
        mask: Visibility mask of pos - array of shape (pred_len, #joint)
    Output:
        errorPose:
    """

    gt_i_global = np.copy(GT)

    if dataset_name == "posetrack":
        mask = np.repeat(mask, 2, axis=-1)
        errorPose = np.power(gt_i_global - pred, 2) * mask
        # get sum on joints and remove the effect of missing joints by averaging on visible joints
        errorPose = np.sqrt(np.divide(np.sum(errorPose, 1), np.sum(mask, axis=1)))
        where_are_NaNs = np.isnan(errorPose)
        errorPose[where_are_NaNs] = 0
    else:  # 3dpw
        errorPose = np.power(gt_i_global - pred, 2)
        errorPose = np.sum(errorPose, 1)
        errorPose = np.sqrt(errorPose)
    return errorPose


def VAM(GT, pred, occ_cutoff, pred_visib):
    """
    Visibility Aware Metric
    Inputs:
        GT: Ground truth data - array of shape (pred_len, #joint*(2D/3D))
        pred: Prediction data - array of shape (pred_len, #joint*(2D/3D))
        occ_cutoff: Maximum error penalty
        pred_visib: Predicted visibilities of pose, array of shape (pred_len, #joint)
    Output:
        seq_err:
    """
    pred_visib = np.repeat(pred_visib, 2, axis=-1)
    # F = 0
    seq_err = []
    if type(GT) is list:
        GT = np.array(GT)
    GT_mask = np.where(abs(GT) < 0.5, 0, 1)

    for frame in range(GT.shape[0]):
        f_err = 0
        N = 0
        for j in range(0, GT.shape[1], 2):
            if GT_mask[frame][j] == 0:
                if pred_visib[frame][j] == 0:
                    dist = 0
                elif pred_visib[frame][j] == 1:
                    dist = occ_cutoff
                    N += 1
            elif GT_mask[frame][j] > 0:
                N += 1
                if pred_visib[frame][j] == 0:
                    dist = occ_cutoff
                elif pred_visib[frame][j] == 1:
                    d = np.power(GT[frame][j:j + 2] - pred[frame][j:j + 2], 2)
                    d = np.sum(np.sqrt(d))
                    dist = min(occ_cutoff, d)
            f_err += dist

        if N > 0:
            seq_err.append(f_err / N)
        else:
            seq_err.append(f_err)
        # if f_err > 0:
        # F += 1
    return np.array(seq_err)


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


def draw_keypoints_posetrack(outputs, image_in=None):
    edges = [(0, 1), (0, 2), (0, 3), (2, 4), (3, 5), (4, 6), (5, 7), (2, 8), (3, 9), (8, 9), (8, 10), (9, 11), (10, 12),
             (11, 13)]

    image = np.zeros((1080, 1920, 3))

    keypoints = outputs

    keypoints = keypoints.reshape(-1, 2)
    print(keypoints.shape)
    for p in range(keypoints.shape[0]):
        # draw the keypoints
        if (not (keypoints[p, 0] <= 0 and keypoints[p, 1] <= 0)):
            cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                       3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            cv2.putText(image, f"{p}", (int(keypoints[p, 0] + 10), int(keypoints[p, 1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1)

    for ie, e in enumerate(edges):
        # get different colors for the edges
        rgb = matplotlib.colors.hsv_to_rgb([
            ie / float(len(edges)), 1.0, 1.0
        ])
        rgb = rgb * 255
        # print(keypoints[e, 0][0])
        # join the keypoint pairs to draw the skeletal structure
        if not ((keypoints[e, 0][0] <= 0 or keypoints[e, 1][0] <= 0) or (
                keypoints[e, 0][1] <= 0 or keypoints[e, 1][1] <= 0)):
            cv2.line(image, (keypoints[e, 0][0], keypoints[e, 1][0]),
                     (keypoints[e, 0][1], keypoints[e, 1][1]),
                     tuple(rgb), 2, lineType=cv2.LINE_AA)

    return image


def draw_keypoints_op(outputs):
    edges = [(13, 15), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (8, 10),
             (7, 9),
             (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]

    #     print(outputs.shape)
    image = np.zeros((1080, 1920, 3))
    for i in range(len(outputs)):
        keypoints = outputs[i]
        keypoints = keypoints.reshape(-1, 2)
        for p in range(keypoints.shape[0]):
            if (not (keypoints[p, 0] == 0 and keypoints[p, 1] == 0)):
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                           3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        for ie, e in enumerate(edges):
            # get different colors for the edges
            rgb = matplotlib.colors.hsv_to_rgb([
                ie / float(len(edges)), 1.0, 1.0
            ])
            rgb = rgb * 255

            if not ((keypoints[e, 0][0] == 0 and keypoints[e, 1][0] == 0) or (
                    keypoints[e, 0][1] == 0 and keypoints[e, 1][1] == 0)):
                cv2.line(image, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                         (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                         tuple(rgb), 2, lineType=cv2.LINE_AA)
        # else:
        #     continue
    #     image=cv2.resize(image,(1280,720))
    return image


def draw_keypoints(outputs):
    edges = [(13, 15), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (8, 10),
             (7, 9),
             (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]

    #     print(outputs.shape)
    image = np.zeros((1080, 1920, 3))
    for i in range(len(outputs)):
        keypoints = outputs[i]
        keypoints = keypoints.reshape(-1, 2)
        for p in range(keypoints.shape[0]):
            if (not (keypoints[p, 0] == 0 and keypoints[p, 1] == 0)):
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                           3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        for ie, e in enumerate(edges):
            # get different colors for the edges
            rgb = matplotlib.colors.hsv_to_rgb([
                ie / float(len(edges)), 1.0, 1.0
            ])
            rgb = rgb * 255

            if not ((keypoints[e, 0][0] == 0 and keypoints[e, 1][0] == 0) or (
                    keypoints[e, 0][1] == 0 and keypoints[e, 1][1] == 0)):
                cv2.line(image, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                         (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                         tuple(rgb), 2, lineType=cv2.LINE_AA)

    return image


def draw_keypoints_op(outputs, image_in=None):
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (8, 12), (9, 10), (10, 11),
             (12, 13), (13, 14), (11, 24), (11, 22), (22, 23), (14, 21), (14, 19), (19, 20), (0, 15), (0, 16), (15, 17),
             (16, 18)]

    image = np.zeros((1080, 1920, 3))
    #     for i in range(len(outputs)):
    keypoints = outputs

    keypoints = keypoints.reshape(-1, 2)
    print(keypoints.shape)
    for p in range(keypoints.shape[0]):
        # draw the keypoints
        if (not (keypoints[p, 0] <= 0 and keypoints[p, 1] <= 0)):
            cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                       3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    #               cv2.putText(image, f"{p}", (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    for ie, e in enumerate(edges):
        # get different colors for the edges
        rgb = matplotlib.colors.hsv_to_rgb([
            ie / float(len(edges)), 1.0, 1.0
        ])
        rgb = rgb * 255

        # join the keypoint pairs to draw the skeletal structure
        if not ((keypoints[e, 0][0] <= 0 and keypoints[e, 1][0] <= 0) or (
                keypoints[e, 0][1] <= 0 and keypoints[e, 1][1] <= 0)):
            cv2.line(image, (keypoints[e, 0][0], keypoints[e, 1][0]),
                     (keypoints[e, 0][1], keypoints[e, 1][1]),
                     tuple(rgb), 2, lineType=cv2.LINE_AA)

    return image


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