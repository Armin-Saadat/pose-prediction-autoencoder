from torch import nn


class Encoder(nn.Module):
    def __init__(self, args, input_size, dropout=0):
        super().__init__()

        self.args = args
        self.rnns = nn.LSTM(input_size=input_size, hidden_size=args.hidden_size, num_layers=self.args.n_layers,
                            dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        encoder_src = self.dropout(src)
        outputs, (hidden, cell) = self.rnn(encoder_src)
        return hidden, cell
