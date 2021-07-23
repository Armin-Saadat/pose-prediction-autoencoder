from torch import nn


class Encoder(nn.Module):
    def __init__(self, args, input_size):
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(self.args.dropout_encoder)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=self.args.hidden_size, num_layers=self.args.n_layers,
                           dropout=0)

    def forward(self, src):
        encoder_src = self.dropout(src)
        outputs, (hidden, cell) = self.rnn(encoder_src)
        return hidden, cell
