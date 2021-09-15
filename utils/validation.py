from torch.autograd import Variable
import torch
import numpy as np


def validate_model(val_loader, model, args, best_acc):
    print('[INFO] Validating the model')
    # Evaluate model in eval mode
    model.eval()

    val_seg_scores = []

    for i, (dsm, rgb, label) in enumerate(val_loader):
        val_dsm_inputs = Variable(dsm.to(args.device))
        val_rgb_inputs = Variable(rgb.to(args.device))
        val_labels = Variable(label.to(args.device))

        print('[PROGRESS] Processing images: %i of %i    ' % (i + 1, len(val_loader)), end='\r')

        val_seg_outputs = model(val_dsm_inputs, val_rgb_inputs)

        _, val_preds = torch.max(val_seg_outputs, 1)

        val_seg_scores.append(np.mean((val_preds == val_labels).data.cpu().numpy()))

    global_acc = np.mean(val_seg_scores)

    if global_acc > best_acc:
        best_acc = global_acc
    print('\r[INFO] Validation has been completed')
    print("[INFO] Validation Seg_Glob_Acc : %.3f, current best accuracy : %.3f " % (global_acc, best_acc))
    return global_acc
