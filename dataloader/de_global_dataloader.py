import torch
import pandas as pd
from ast import literal_eval


class DE_Global_DataLoader(torch.utils.data.Dataset):
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

        return tuple(outputs)


def data_loader_de_global(args, data, fname):
    dataset = DE_Global_DataLoader(args, data, fname)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory)
    return dataloader
