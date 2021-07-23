import torch.nn as nn

from models.decoder import PoseDecoder
from models.encoder import Encoder


class LSTMVel3dpw(nn.Module):
    def __init__(self, args):
        super(LSTMVel3dpw, self).__init__()

        self.pose_encoder = Encoder(args=self.args, input_size=39, dropout=0)
        self.vel_encoder = Encoder(args=self.args, input_size=39, dropout=0)

        self.vel_decoder = PoseDecoder(args=self.args, dropout=0, out_features=39, input_size=39)
        self.args = args

    def forward(self, pose=None, vel=None):
        outputs = []

        (hidden_vel, cell_vel) = self.vel_encoder(vel.permute(1, 0, 2))
        hidden_vel = hidden_vel.squeeze(0)
        cell_vel = cell_vel.squeeze(0)

        (hidden_pose, cell_pose) = self.pose_encoder(pose.permute(1, 0, 2))
        hidden_pose = hidden_pose.squeeze(0)
        cell_pose = cell_pose.squeeze(0)

        VelDec_inp = vel[:, -1, :]

        hidden_dec = hidden_pose + hidden_vel
        cell_dec = cell_pose + cell_vel

        outputs.append(self.vel_decoder(VelDec_inp, hidden_dec, cell_dec))
        return tuple(outputs)
