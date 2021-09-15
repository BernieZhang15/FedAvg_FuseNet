import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset_utils_labels import Dataset
from models.SegUNet import SegmentationUNet

def convert_backward(image):
    y = np.zeros((256, 256, 3))
    index1 = torch.where(image == 0)
    y[index1[1], index1[2]] = [0, 0, 255]
    index2 = torch.where(image == 1)
    y[index2[1], index2[2]] = [255, 255, 255]
    index3 = torch.where(image == 2)
    y[index3[1], index3[2]] = [255, 0, 0]
    index4 = torch.where(image == 3)
    y[index4[1], index4[2]] = [255, 255, 0]
    index5 = torch.where(image == 4)
    y[index5[1], index5[2]] = [0, 255, 0]
    index6 = torch.where(image == 5)
    y[index6[1], index6[2]] = [0, 255, 255]
    return y

if __name__ == '__main__':
    print('[INFO] Test and display the performance')

    device = torch.device('cuda:0')
    path = "checkpoints/experiment/best_model.pth.tar"
    model = SegmentationUNet(6).to(device)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset_display = Dataset("DSM", "RGB", "LABEL", 4)
    dataloader_display = DataLoader(dataset_display, batch_size=1, shuffle=False)

    test_seg_scores = []

    for i, (dsm, rgb, label) in enumerate(dataloader_display):
        dsm_inputs = dsm.to(device)
        rgb_inputs = rgb.to(device)
        label = label.to(device)

        print('[PROGRESS] Processing images: %i of %i    ' % (i + 1, len(dataloader_display)), end='\r')

        seg_outputs = model(dsm_inputs, rgb_inputs)

        _, test_preds = torch.max(seg_outputs, 1)

        test_seg_scores.append(np.mean((test_preds == label).data.cpu().numpy()))

        # convert back to image
        test_preds = convert_backward(test_preds.cpu())
        cv2.imwrite('result\\predict_{}.jpg'.format(i), test_preds)
        label = convert_backward(label.cpu())
        cv2.imwrite('result\\groundtruth_{}.jpg'.format(i), label)

    global_acc = np.mean(test_seg_scores)

    print('\r[INFO] Testing is completed')
    print("[INFO] Test Seg_Glob_Acc : %.3f" % (global_acc))

