import torch.nn as nn

from models.decoder import PoseDecoder, MaskDecoder
from models.encoder import Encoder


class LSTMVelPosetrack(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pose_encoder = Encoder(args=self.args, input_size=28, dropout=0)
        self.vel_encoder = Encoder(args=self.args, input_size=28, dropout=0)
        self.vel_decoder = PoseDecoder(args=self.args, dropout=0, out_features=28, input_size=28)

        self.mask_encoder = Encoder(args=self.args, input_size=14, dropout=0)
        self.mask_decoder = MaskDecoder(args=self.args, dropout=0, out_features=14, input_size=14)

    def forward(self, pose=None, vel=None, mask=None):
        outputs = []

        (hidden_vel, cell_vel) = self.vel_encoder(vel.permute(1, 0, 2))
        hidden_vel = hidden_vel.squeeze(0)
        cell_vel = cell_vel.squeeze(0)

        (hidden_pose, cell_pose) = self.pose_encoder(pose.permute(1, 0, 2))
        hidden_pose = hidden_pose.squeeze(0)
        cell_pose = cell_pose.squeeze(0)

        (hidden_mask, cell_mask) = self.mask_encoder(mask.permute(1, 0, 2))
        hidden_mask = hidden_mask.squeeze(0)
        cell_mask = cell_mask.squeeze(0)

        VelDec_inp = vel[:, -1, :]
        MaskDec_inp = mask[:, -1, :]

        hidden_dec = hidden_pose + hidden_vel
        cell_dec = cell_pose + cell_vel
        outputs.append(self.vel_decoder(VelDec_inp, hidden_dec, cell_dec))
        outputs.append(self.mask_decoder(MaskDec_inp, hidden_mask, cell_mask))
        return outputs
