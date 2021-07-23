import torch
import numpy as np


def ADE_c(pred_pose, target_pose):
    b, n, p = pred_pose.size()[0], pred_pose.size()[1], pred_pose.size()[2]
    pred_pose = torch.reshape(pred_pose, (b, n, int(p / 2), 2))
    target_pose = torch.reshape(target_pose, (b, n, int(p / 2), 2))
    displacement = torch.sqrt(
        (pred_pose[:, :, :, 0] - target_pose[:, :, :, 0]) ** 2 + (pred_pose[:, :, :, 1] - target_pose[:, :, :, 1]) ** 2)
    ade = torch.mean(torch.mean(displacement, dim=1))
    return ade


def FDE_c(pred_pose, target_pose):
    b, n, p = pred_pose.size()[0], pred_pose.size()[1], pred_pose.size()[2]
    pred_pose = torch.reshape(pred_pose, (b, n, int(p / 2), 2))
    target_pose = torch.reshape(target_pose, (b, n, int(p / 2), 2))
    displacement = torch.sqrt(
        (pred_pose[:, -1, :, 0] - target_pose[:, -1, :, 0]) ** 2 + (
                pred_pose[:, -1, :, 1] - target_pose[:, -1, :, 1]) ** 2)
    fde = torch.mean(torch.mean(displacement, dim=1))
    return fde


def ADE_3d(pred, true):
    b, n, p = pred.size()[0], pred.size()[1], pred.size()[2]
    pred = torch.reshape(pred, (b, n, int(p / 3), 3))
    true = torch.reshape(true, (b, n, int(p / 3), 3))
    displacement = torch.sqrt((pred[:, :, :, 0] - true[:, :, :, 0]) ** 2 + (pred[:, :, :, 1] - true[:, :, :, 1]) ** 2)
    ade = torch.mean(torch.mean(displacement, dim=1))
    return ade


def FDE_3d(pred, true):
    b, n, p = pred.size()[0], pred.size()[1], pred.size()[2]
    pred = torch.reshape(pred, (b, n, int(p / 3), 3))
    true = torch.reshape(true, (b, n, int(p / 3), 3))
    displacement = torch.sqrt(
        (pred[:, -1, :, 0] - true[:, -1, :, 0]) ** 2 + (pred[:, -1, :, 1] - true[:, -1, :, 1]) ** 2)
    fde = torch.mean(torch.mean(displacement, dim=1))
    return fde


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
    return np.array(seq_err)


def mask_accuracy(preds, trues):
    zeros = torch.zeros_like(preds)
    ones = torch.ones_like(preds)
    preds = torch.where(preds > 0.5, ones, zeros)
    # n_zeros = torch.sum((preds == trues) * (preds == 0))
    # n_ones = torch.sum((preds == trues) * (preds == 1))
    # return (n_ones + n_zeros) / torch.numel(preds)
    return torch.sum(preds == trues) / torch.numel(preds)
