import os
import torch
import numpy as np
from utils.options import args_parse
from models.FuseNet import FederatedNet
from utils.dataset_utils_labels import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = args_parse()
    args.device = torch.device("cuda:0")
    states = torch.load(args.model_path)

    model = FederatedNet(args.num_labels).to(args.device)
    model.load_state_dict(states["state_dict"])
    model.eval()

    print('[INFO] Dataset is being processed')
    dataset_test = Dataset("DSM", "RGB", "LABEL", 4)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
    print("[INFO] Data loaders for Remote Sensing dataset have been created")

    seg_scores = []
    for index, (dsm, rgb, label) in enumerate(dataloader_test):
        dsm_input = dsm.to(args.device)
        rgb_input = rgb.to(args.device)
        seg_label = label.to(args.device)

        seg_outputs = model(dsm_input, rgb_input)

        _, val_preds = torch.max(seg_outputs, 1)
        print('[PROGRESS] Processing images: %i of %i    ' % (index + 1, len(dataloader_test)), end='\r')

        seg_scores.append(np.mean((val_preds == seg_label).data.cpu().numpy()))

    print(np.mean(seg_scores))

    # plot loss curve
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(range(len(states['train_loss_hist'])), states['train_loss_hist'])
    plt.ylabel('train_loss')
    plt.subplot(1, 2, 2)
    plt.plot(range(len(states['val_seg_acc_hist'])), states['val_seg_acc_hist'])
    plt.ylabel('validation_accuracy')
    plt.show()
    plt.savefig('save/fed results.png')
