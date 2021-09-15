import copy
import torch
from torch.utils.data import DataLoader, Dataset
from utils.cross_entropy_loss import CrossEntropy2d
from utils.focol_tversky_loss import TverskyCrossEntropyDiceWeightedLoss


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        dsm, rgb, label = self.dataset[self.idxs[item]]
        return dsm, rgb, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, user_id=None, epoch=None, mu=0):
        self.args = args

        if args.loss_func == "Tversky":
            self.loss_func = TverskyCrossEntropyDiceWeightedLoss(args.num_labels, args.device)
        else:
            self.loss_func = CrossEntropy2d()
        self.proximal_criterion = torch.nn.MSELoss(reduction='mean')
        self.user_id = user_id
        self.epoch = epoch
        self.mu = mu
        self.dl_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        global_model = copy.deepcopy(net)

        epoch_loss = []
        for iter in range(self.epoch):
            batch_loss = []
            for batch_idx, (dsm, rgb, labels) in enumerate(self.dl_train):
                dsm_inputs = dsm.to(self.args.device)
                rgb_inputs = rgb.to(self.args.device)
                seg_labels = labels.to(self.args.device)

                net.zero_grad()
                log_probs = net(dsm_inputs, rgb_inputs)

                if self.args.algorithm == 'FedProx' and self.mu != 0:
                    proximal_term = 0.0
                    for w, w_global in zip(net.parameters(), global_model.parameters()):
                        proximal_term += (w - w_global).norm(2)
                    loss = self.loss_func(log_probs, seg_labels.long()) + (self.mu / 2) * proximal_term
                else:
                    loss = self.loss_func(log_probs, seg_labels.long())

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print('User {} Loss: {}'.format(self.user_id, avg_loss))
        return net.state_dict(), avg_loss

