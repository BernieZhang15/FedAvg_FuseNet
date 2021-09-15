import random
import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from utils.sampling import split_data
from utils.options import args_parse
from models.Update import LocalUpdate
from models.FuseNet import FederatedNet
from models.SegUNet import SegmentationUNet
from utils.dataset_utils_labels import Dataset
from utils.validation import validate_model
from utils.save_checkpoints import save_checkpoints
from models.Fed import FedAvg
from utils.local_epochs_utils import generateLocalEpochs
matplotlib.use('Agg')


def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def reset_histories_and_losses(states):
    """
    Resets train and val histories for accuracy and the loss.
    """
    states['epoch'] = 0

    states['train_loss_hist'] = []

    states['val_seg_acc_hist'] = []

    states['best_val_seg_acc'] = 0.0


def load_checkpoint(path, model, states):
    if os.path.isfile(path):
        print('[PROGRESS] Loading checkpoint: {}'.format(path), end="", flush=True)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        states.update({key : value for key, value in checkpoint.items() if key not in ['state_dict']})
        print('\r[INFO] Checkpoint has been loaded: {}'.format(path))
        print('[INFO] History lists have been loaded')
        print('[INFO] Resuming from epoch {}'.format(checkpoint['epoch'] + 1))

        del checkpoint
        torch.cuda.empty_cache()
    else:
        raise FileNotFoundError('Checkpoint file not found: %s' % path)


if __name__ == '__main__':
    # parse args
    seed_torch()
    args = args_parse()
    args.device = torch.device('cuda:0')

    # load dataset and split users
    print('[INFO] Dataset is being processed')
    dataset_train = Dataset("DSM", "RGB", "LABEL", args.dataset_train)

    dataset_val = Dataset("DSM", "RGB", "LABEL", args.dataset_val)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
    print("[INFO] Data loaders for Remote Sensing dataset have been created")
    print("[INFO] Training uses {} optimization algorithm".format(args.algorithm))
    # sample users
    dict_users = split_data(dataset_train, args.num_users)

    # build model
    if args.model == "FuseNet":
        model = FederatedNet(args.num_labels).to(args.device)
    elif args.model == "SegUNet":
        model = SegmentationUNet(args.num_labels).to(args.device)

    # training
    states = dict()
    reset_histories_and_losses(states)

    if args.load_checkpoint is not None:
        load_checkpoint(args.load_checkpoint, model, states)

    # copy weights
    w_glob = model.state_dict()

    m = max(int(args.frac * args.num_users), 1)
    hetero_epoch_list = generateLocalEpochs(percentage=args.percentage, size=m, max_epochs=args.local_eps)

    for iter in range(states['epoch'], args.num_epochs):

        model.train()
        loss_locals = []
        w_locals = []

        # update learning rate
        if (iter + 1) % 20 == 0:
            args.lr *= 0.9

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        print("\nStart No.{} epoch training".format(iter + 1))
        print("System heterogeneity set to {}% stragglers.".format(args.percentage))
        print("Pick {} random clients per round.".format(m))

        if args.algorithm == "FedAvg":
            stragglers_indices = np.argwhere(hetero_epoch_list < args.local_eps)
            hetero_epoch_list = np.delete(hetero_epoch_list, stragglers_indices)
            idxs_users = np.delete(idxs_users, stragglers_indices)

        for num, idx in enumerate(idxs_users):
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx],
                                user_id=idx, epoch=hetero_epoch_list[num], mu=args.mu)
            w, loss = local.train(net=copy.deepcopy(model).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        model.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        states['train_loss_hist'].append(loss_avg)
        print('Epoch %d , Average loss %.3f' % (iter + 1, loss_avg))

        global_acc = validate_model(dataloader_val, model, args, states['best_val_seg_acc'])
        states['val_seg_acc_hist'].append(global_acc)

        is_best = global_acc > states['best_val_seg_acc']
        if (iter + 1) % args.save_epoch_freq == 0 or is_best:
            states['epoch'] = iter
            if is_best:
                states['best_val_seg_acc'] = global_acc
            states['state_dict'] = model.state_dict()
            save_checkpoints(states, args, is_best)

    # plot loss curve
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(range(len(states['train_loss_hist'])), states['train_loss_hist'])
    plt.ylabel('train_loss')
    plt.subplot(1, 2, 2)
    plt.plot(range(len(states['val_seg_acc_hist'])), states['val_seg_acc_hist'])
    plt.ylabel('validation_accuracy')
    plt.show()
    plt.savefig('./save/Fed with {} epochs, {} Users and lr_{}.png'.format(args.num_epochs, args.num_users, args.lr))
