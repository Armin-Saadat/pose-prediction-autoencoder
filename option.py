class Option:
    def __init__(self, epoch=200, batch_size=128, learning_rate=0.01, lr_decay_rate=0.25, load_ckpt=None):
        self.batch_size = batch_size
        self.num_workers = 1
        self.epochs = epoch
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.loader_shuffle = False
        self.pin_memory = False
        self.device = 'cuda'
        self.hidden_size = 1000
        self.hardtanh_limit = 100
        self.dataset_name = 'posetrack'  # choices=['posetrack', '3dpw']
        self.model_name = 'lstm_vel'  # choices=['lstm_vel', 'disentangling']
        self.input = 16
        self.output = 14
        self.save_folder = 'snapshots'
        self.save_freq = 50
        self.load_ckpt = load_ckpt
        self.stride = self.input
        self.skip = 1
