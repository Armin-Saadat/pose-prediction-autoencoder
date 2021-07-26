from torch import nn
import torch


class VelDecoder(nn.Module):
    def __init__(self, args, out_features, input_size=28):
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(self.args.dropout_pose_decoder)
        rnns = [
            nn.LSTMCell(input_size=input_size if i == 0 else args.hidden_size, hidden_size=args.hidden_size).cuda() for
            i in range(args.n_layers)]
        self.rnns = nn.Sequential(*rnns)
        self.fc_out = nn.Linear(in_features=args.hidden_size, out_features=out_features)
        self.hardtanh = nn.Hardtanh(min_val=-1 * args.hardtanh_limit, max_val=args.hardtanh_limit, inplace=False)

    def forward(self, inputs, hiddens, cells):
        dec_inputs = self.dropout(inputs)
        if len(hiddens.shape) < 3 or len(cells.shape) < 3:
            hiddens = torch.unsqueeze(hiddens, 0)
            cells = torch.unsqueeze(cells, 0)
        outputs = torch.tensor([], device=self.args.device)
        for j in range(self.args.output):
            for i, rnn in enumerate(self.rnns):
                if i == 0:
                    hiddens[i], cells[i] = rnn(dec_inputs, (hiddens.clone()[i], cells.clone()[i]))
                else:
                    hiddens[i], cells[i] = rnn(hiddens.clone()[i - 1], (hiddens.clone()[i], cells.clone()[i]))
            output = self.hardtanh(self.fc_out(hiddens.clone()[-1]))
            dec_inputs = output.detach()
            outputs = torch.cat((outputs, output.unsqueeze(1)), dim=1)
        return outputs


class MaskDecoder(nn.Module):
    def __init__(self, args, out_features, input_size=14):
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(self.args.dropout_mask_decoder)
        rnns = [
            nn.LSTMCell(input_size=input_size if i == 0 else args.hidden_size, hidden_size=args.hidden_size).cuda() for
            i in range(args.n_layers)]
        self.rnns = nn.Sequential(*rnns)
        self.fc_out = nn.Linear(in_features=args.hidden_size, out_features=out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, hiddens, cells):
        dec_input = self.dropout(inputs)
        if len(hiddens.shape) < 3 or len(cells.shape) < 3:
            hiddens = torch.unsqueeze(hiddens, 0)
            cells = torch.unsqueeze(cells, 0)
        outputs = torch.tensor([], device=self.args.device)
        for j in range(self.args.output):
            for i, rnn in enumerate(self.rnns):
                if i == 0:
                    hiddens[i], cells[i] = rnn(dec_input, (hiddens.clone()[i], cells.clone()[i]))
                else:
                    hiddens[i], cells[i] = rnn(hiddens.clone()[i - 1], (hiddens.clone()[i], cells.clone()[i]))
            output = self.sigmoid(self.fc_out(hiddens.clone()[-1]))
            dec_input = output.detach()
            outputs = torch.cat((outputs, output.unsqueeze(1)), dim=1)
        return outputs
