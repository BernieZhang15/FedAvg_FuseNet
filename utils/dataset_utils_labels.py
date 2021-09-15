import os
import torch
import numpy as np
from PIL import Image
import tifffile as tiff
from torchvision import transforms
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, img_dir1, img_dir2, img_dir3, flag):
        super(Dataset, self).__init__()
        self.transform = transforms.Compose([transforms.Grayscale(1)])
        self.transform2 = transforms.Compose([transforms.ToTensor()])
        str = "Train"
        if flag == 2:
            str = "Validation"
        elif flag == 3:
            str = "Test"
        elif flag == 4:
            str = "Debug"
        self.dsm = self.read_jpg(os.path.join("..", "Data","Potsdam", str, img_dir1))
        self.rgb = self.read_tif(os.path.join("..", "Data", "Potsdam", str, img_dir2), False)
        self.label = self.read_tif(os.path.join("..", "Data","Potsdam", str, img_dir3), True)

    def read_jpg(self, path):
        img_dir = []
        files = os.listdir(path)
        for file in files:
            img = Image.open(path + '/' + file)
            img = self.transform2(img)
            img_dir.append(img)
        return img_dir

    def read_tif(self, path, flag):
        img_dir = []
        files = os.listdir(path)
        for file in files:
            img = tiff.imread(path + '/' + file)
            if flag:
                img = Image.fromarray(img)
                img = self.transform(img)
                img = np.array(img)
                img = torch.tensor(img)
                img = self.convert_forward(img)
            else:
                img = self.transform2(img)
            img_dir.append(img)
        return img_dir

    def convert_forward(self, image):
        y = torch.zeros((256, 256))
        index1 = torch.where(image == 226)
        y[index1[0], index1[1]] = 0
        index2 = torch.where(image == 76)
        y[index2[0], index2[1]] = 1
        index3 = torch.where(image == 179)
        y[index3[0], index3[1]] = 2
        index4 = torch.where(image == 150)
        y[index4[0], index4[1]] = 3
        index5 = torch.where(image == 29)
        y[index5[0], index5[1]] = 4
        index6 = torch.where(image == 255)
        y[index6[0], index6[1]] = 5
        return y

    def __len__(self):
        return len(self.dsm)

    def __getitem__(self, index):
        return self.dsm[index], self.rgb[index], self.label[index]
