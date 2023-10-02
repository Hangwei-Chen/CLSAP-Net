import torch
import torchvision
import folders
import time
class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, SR_path, CP_path, OV_path, img_indx, patch_size, patch_num, batch_size=1, istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain


        if istrain:
            flip=1
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
            # Test transforms
        else:
            flip=0
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])



        self.data = folders.Folder(
            SR_root=SR_path, CP_root=CP_path, OV_root=OV_path, index=img_indx, transform=transforms, patch_num=patch_num, patch_size=patch_size, flip=flip)

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False)
        return dataloader

