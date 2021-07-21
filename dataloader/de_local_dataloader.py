import torch
import pandas as pd
from ast import literal_eval


class DE_Local_DataLoader(torch.utils.data.Dataset):
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
        local_obs = self.create_local_var(obs)
        obs_speed = (obs[1:, 2:] - obs[:-1, 2:])

        true = torch.tensor([seq.Future_Pose[i] for i in range(0, self.args.output, self.args.skip)])
        local_true = self.create_local_var(true)
        true_speed = torch.cat(((local_true[0] - local_obs[-1]).unsqueeze(0), local_true[1:] - local_true[:-1]))
        outputs.append(obs_speed)
        outputs.append(true_speed)
        outputs.append(local_obs)
        outputs.append(local_true)

        if self.fname == "posetrack_":
            obs_mask = torch.tensor([seq.Mask[i] for i in range(0, self.args.output, self.args.skip)])
            true_mask = torch.tensor([seq.Future_Mask[i] for i in range(0, self.args.output, self.args.skip)])
            outputs.append(obs_mask)
            outputs.append(true_mask)
        return tuple(outputs)

    def create_local_var(self, data):
        neck_joint_data = data[:, :2]  # [16,2]
        other_joints_data = data[:, 2:]  # [16,26]
        for i, val in enumerate(other_joints_data):
            for j in range(13):
                val[2*j: 2*(j+1)] = torch.sub(val[2*j: 2*(j+1)], neck_joint_data[i])
        return other_joints_data


def data_loader_de_local(args, data, fname):
    dataset = DE_Local_DataLoader(args, data, fname)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory)
    return dataloader
