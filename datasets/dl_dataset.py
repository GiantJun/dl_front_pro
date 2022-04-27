from os.path import join, splitext, dirname
from os import listdir, environ
from torch.utils.data import Dataset
from PIL import Image
from torch import cat
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

class DLDataset(Dataset):
    """自定义 Dataset 类"""
    def __init__(self, csv_path, transform=None):
        self.transform = transform
        base_dir = dirname(csv_path)
        self.data_dir = join(base_dir, 'cleaned_data')
        self.data_frame = pd.read_csv(csv_path)

    def __len__(self):
        """返回 dataset 大小"""
        return len(self.data_frame)

    def set_transforms(self, tf):
        self.transform = tf

    def __getitem__(self, idx):
        """根据 idx 从 图片名字-类别 列表中获取相应的 image 和 label"""
        img_path = join(self.data_dir, self.data_frame.iat[idx, 0].lower())
        image = Image.open(img_path)
        target = self.data_frame.iat[idx, 1]

        if self.transform:
            sample = self.transform(image)

        return sample, target, img_path

# if __name__ == '__main__':
#     # 计算灰度图时使用
#     tf = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])
#     select_list=list(range(0,3))
#     dataset = DLDataset('/media/huangyujun/disk/data/深度学习前沿大作业_超声数据集/内部数据集/Original_train_dataset/label.csv',transform=tf)
#     dataloader = DataLoader(dataset=dataset, batch_size=int(len(dataset)), shuffle=False, num_workers=4)
#     img_l = []
#     for imgs, label, img_dir in dataloader:
#         imgs = imgs.numpy()
#         print(imgs.shape)
#         for i in range(len(select_list)):
#             # 对于单通道时计算mean和std使用
#             img_l.append(imgs[:,i,:,:])
#         all = np.vstack(img_l)
#         print(all.shape)
#         mean = np.mean(all)
#         std = np.std(all)
#     print(mean, std)