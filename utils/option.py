import argparse


class Option:
    def __init__(self, epochs=200, batch_size=128, learning_rate=0.01, lr_decay_rate=0.25, load_ckpt=None):
        self.batch_size = batch_size
        self.num_workers = 1
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.loader_shuffle = False
        self.pin_memory = False
        self.device = 'cuda'
        self.hidden_size = 1000
        self.hardtanh_limit = 100
        self.dataset_name = 'posetrack'  # choices=['posetrack', '3dpw']
        self.model_name = 'lstm_vel'  # choices=['lstm_vel', 'de_local', 'de_global']
        self.input = 16
        self.output = 14
        self.save_folder = '../snapshots'
        self.save_freq = 50
        self.load_ckpt = load_ckpt
        self.stride = self.input
        self.skip = 1
        self.n_layers = 1
        self.dropout_encoder = 0
        self.dropout_pose_decoder = 0
        self.dropout_mask_decoder = 0


def parse_option(model_name, dataset_name):
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=80, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.25, help='decay rate for learning rate')
    parser.add_argument('--loader_shuffle', type=bool, default=False)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--hidden_size', type=int, default=1000)
    parser.add_argument('--hardtanh_limit', type=int, default=100)
    parser.add_argument('--input', type=int, default=16)
    parser.add_argument('--output', type=int, default=14)
    # parser.add_argument('--model_name', type=str, default='lstm_vel', choices=['lstm_vel', 'de_global', 'de_local'])
    # parser.add_argument('--dataset_name', type=str, default='posetrack', choices=['posetrack', '3dpw'])
    parser.add_argument('--save_folder', type=str, default='../snapshots')
    parser.add_argument('--save_freq', type=int, default=198)
    parser.add_argument('--load_ckpt', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dropout_encoder', type=float, default=0)
    parser.add_argument('--dropout_pose_decoder', type=float, default=0)
    parser.add_argument('--dropout_mask_decoder', type=float, default=0)
    opt = parser.parse_args()
    opt.stride = opt.input
    opt.skip = 1
    if opt.save_freq == 198:
        opt.save_freq = opt.epochs
    opt.model_name = model_name
    opt.dataset_name = dataset_name
    return opt
