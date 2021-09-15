import torch
import torch.nn as nn
import torch.nn.functional as F


class FederatedNet(nn.Module):
    def __init__(self, num_labels):
        super(FederatedNet, self).__init__()

        batchNorm_momentum = 0.1

        # DSM encoder
        self.conv11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.dropout_3 = nn.Dropout(p=0.5)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.dropout_4 = nn.Dropout(p=0.5)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        # RGB encoder
        self.conv11r = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.bn11r = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12r = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12r = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv21r = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21r = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22r = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22r = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31r = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31r = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32r = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32r = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33r = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33r = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.dropout_3r = nn.Dropout(p=0.5)

        self.conv41r = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41r = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42r = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42r = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43r = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43r = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.dropout_4r = nn.Dropout(p=0.5)

        self.conv51r = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51r = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52r = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52r = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53r = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53r = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.dropout_5r = nn.Dropout(p=0.5)

        # decoder
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.dropout_5d = nn.Dropout(p=0.5)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.dropout_4d = nn.Dropout(p=0.5)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.dropout_3d = nn.Dropout(p=0.5)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, num_labels, kernel_size=3, padding=1)

        print('[INFO] FuseNet model has been created')

    def forward(self,  dsm_inputs, rgb_inputs):
        # DSM Stage 1
        x11 = F.relu(self.bn11(self.conv11(dsm_inputs)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)

        # DSM Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)

        # DSM Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)
        x3p = self.dropout_3(x3p)

        # DSM Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)
        x4p = self.dropout_4(x4p)

        # DSM Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))

        # RGB Stage 1
        y11 = F.relu(self.bn11r(self.conv11r(rgb_inputs)))
        y12 = F.relu(self.bn12r(self.conv12r(y11)))
        y12 = torch.add(y12, x12)
        y1p, id6 = F.max_pool2d(y12, kernel_size=2, stride=2, return_indices=True)

        # RGB Stage 2
        y21 = F.relu(self.bn21r(self.conv21r(y1p)))
        y22 = F.relu(self.bn22r(self.conv22r(y21)))
        y22 = torch.add(y22, x22)
        y2p, id7 = F.max_pool2d(y22, kernel_size=2, stride=2, return_indices=True)

        # RGB Stage 3
        y31 = F.relu(self.bn31r(self.conv31r(y2p)))
        y32 = F.relu(self.bn32r(self.conv32r(y31)))
        y33 = F.relu(self.bn33r(self.conv33r(y32)))
        y33 = torch.add(y33, x33)
        y3p, id8 = F.max_pool2d(y33, kernel_size=2, stride=2, return_indices=True)
        y3p = self.dropout_3r(y3p)

        # RGB Stage 4
        y41 = F.relu(self.bn41r(self.conv41r(y3p)))
        y42 = F.relu(self.bn42r(self.conv42r(y41)))
        y43 = F.relu(self.bn43r(self.conv43r(y42)))
        y43 = torch.add(y43, x43)
        y4p, id9 = F.max_pool2d(y43, kernel_size=2, stride=2, return_indices=True)
        y4p = self.dropout_4r(y4p)

        # RGB Stage 5
        y51 = F.relu(self.bn51r(self.conv51r(y4p)))
        y52 = F.relu(self.bn52r(self.conv52r(y51)))
        y53 = F.relu(self.bn53r(self.conv53r(y52)))
        y53 = torch.add(y53, x53)
        y5p, id10 = F.max_pool2d(y53, kernel_size=2, stride=2, return_indices=True)
        y5p = self.dropout_5r(y5p)

        # Stage 5d
        x5d = F.max_unpool2d(y5p, id10, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))
        x51d = self.dropout_5d(x51d)

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id9, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))
        x41d = self.dropout_4d(x41d)

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id8, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))
        x31d = self.dropout_3d(x31d)

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id7, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id6, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d
