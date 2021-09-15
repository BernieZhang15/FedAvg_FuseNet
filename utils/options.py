from argparse import ArgumentParser


def args_parse():
    parser = ArgumentParser(description='Semantic segmentation')
    parser.add_argument('--num_epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum factor for SGD')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for validation')
    parser.add_argument('--num_labels', type=int, default=6, help='Number of output labels')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='momentum factor for optimizer')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models are saved here')
    parser.add_argument('--name', type=str, default='experiment',
                        help='checkpoints of the current experiment are saved here')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='load checkpoint to resume training')
    parser.add_argument('--frac', type=float, default=0.7, help="fraction of clients chosen to perform: C")
    parser.add_argument('--percentage', type=float, default=0,
                        help="percentage of selected clients to local train fewer than E epochs")
    parser.add_argument('--local_eps', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=5, help="local batch size: B")
    parser.add_argument('--mu', type=float, default=0.4, help="the mu for proximal term in FedProx")
    parser.add_argument('--save_epoch_freq', type=int, default=20,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--model_path', type=str, default='checkpoints/experiment/best_model.pth.tar',
                        help='load model for test_module function')
    parser.add_argument('--dataset_train', type=int, default=1, help="dataset for training")
    parser.add_argument('--dataset_val', type=int, default=2, help="dataset for validation")
    parser.add_argument('--algorithm', type=str, default='FedAvg', help='Optimization algorithm')
    parser.add_argument('--loss_func', type=str, default='Tversky', help='loss function for local update')
    parser.add_argument('--model', type=str, default='SegUNet', help='model you use for training')
    args = parser.parse_args()
    return args
