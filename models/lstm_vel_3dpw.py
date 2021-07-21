import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms


class LSTM_Vel_3dpw(nn.Module):
    def __init__(self, args):
        print("here")
        print(args)
        super(LSTM_Vel_3dpw, self).__init__()

        self.pose_encoder = nn.LSTM(input_size=39, hidden_size=args.hidden_size)
        self.vel_encoder = nn.LSTM(input_size=39, hidden_size=args.hidden_size)

        self.pose_embedding = nn.Sequential(nn.Linear(in_features=args.hidden_size, out_features=39),
                                            nn.ReLU())

        self.vel_decoder = nn.LSTMCell(input_size=39, hidden_size=args.hidden_size)

        self.fc_vel = nn.Linear(in_features=args.hidden_size, out_features=39)

        self.hardtanh = nn.Hardtanh(min_val=-1 * args.hardtanh_limit, max_val=args.hardtanh_limit)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.args = args

    def forward(self, pose=None, vel=None):
        _, (hidden_vel, cell_vel) = self.vel_encoder(vel.permute(1, 0, 2))
        hidden_vel = hidden_vel.squeeze(0)
        cell_vel = cell_vel.squeeze(0)

        _, (hidden_pose, cell_pose) = self.pose_encoder(pose.permute(1, 0, 2))
        hidden_pose = hidden_pose.squeeze(0)
        cell_pose = cell_pose.squeeze(0)

        outputs = []

        vel_outputs = torch.tensor([], device=self.args.device)

        VelDec_inp = vel[:, -1, :]

        hidden_dec = hidden_pose + hidden_vel
        cell_dec = cell_pose + cell_vel
        for i in range(self.args.output // self.args.skip):
            hidden_dec, cell_dec = self.vel_decoder(VelDec_inp, (hidden_dec, cell_dec))
            vel_output = self.hardtanh(self.fc_vel(hidden_dec))
            vel_outputs = torch.cat((vel_outputs, vel_output.unsqueeze(1)), dim=1)
            VelDec_inp = vel_output  # .detach()

        outputs.append(vel_outputs)
        return tuple(outputs)