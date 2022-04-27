# @ FileName: gallbladder_dataset.py
# @ Author: Alexis
# @ Time: 20-11-28 下午9:17
from os.path import join, splitext, basename
from os import listdir, environ
from torch.utils.data import Dataset
from PIL import Image
# import SimpleITK as sitk
from torch import cat
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

# 预先运行本文件计算好的 mean 和 std
mean_RGB = [0.3215154, 0.32233492, 0.3232533]
std_RGB = [0.1990278, 0.19854955, 0.19809964]
mean_gray = 0.3505138
std_gray = 0.21441448

class AltrasoundDataset(Dataset):
    """自定义 Dataset 类"""
    def __init__(self, root, transform=None, select_list=list(range(0,9)),dataset_type='train'):
        """
        输入：root_dir：数据子集的根目录
        dataset_type如果为“test”则会获取测试集数据
        说明：此处的root_dir 需满足如下的格式：
        root_dir/
        |-- class0
        |   |--image_group0
        |   |   |--1.png
        |   |   |--2.png
        |   |   |--3.png
        |   |--image_group2
        |   |   |--1.png
        |   |   |--2.png
        |-- class1
        """
        self.transform = transform
        self.item_list = []
        self.select_list = select_list  # 选择9幅图中的几幅（从0开始计）
        self.class_names = None
        self.class_num = None
        if dataset_type == 'all':
            root_dir = join(root,'train')
            # 训练集测试集下，各类别名默认为一致
            self.class_names = listdir(root_dir)
            self.class_names.sort()
            self.class_num = len(self.class_names)
            for idx, item in enumerate(self.class_names):   # idx 相当于类别标号，此处0-nonstandard,1-standard
                for type in ['train','test']:
                    root_dir = join(root,type)
                    # 训练集测试集下，各类别名默认为一致
                    class_dir = join(root_dir, item)
                    img_dirs = listdir(class_dir)
                    # img_dirs.sort()
                    self.item_list.extend([(join(class_dir, image_dir), idx) for image_dir in img_dirs])
        elif dataset_type in ['train','test']:
            root_dir = join(root,dataset_type)
            self.class_names = listdir(root_dir)
            self.class_names.sort()
            self.class_num = len(self.class_names)
            
            for idx, item in enumerate(self.class_names):   # idx 相当于类别标号，此处0-nonstandard,1-standard
                class_dir = join(root_dir, item)
                img_dirs = listdir(class_dir)
                # img_dirs.sort()
                self.item_list.extend([(join(class_dir, image_dir), idx) for image_dir in img_dirs])
        else:
            raise ValueError('No dataset type {} , dataset type must in [\'train\',\'test\']'.format(dataset_type))
        
        print('0-{} 1-{}'.format(self.class_names[0],self.class_names[1]))
        
    def get_XY(self):
        x, y = zip(*(self.item_list))
        return x, y

    def __len__(self):
        """返回 dataset 大小"""
        return len(self.item_list)

    def set_transforms(self, tf):
        self.transform = tf

    def __getitem__(self, idx):
        """根据 idx 从 图片名字-类别 列表中获取相应的 image 和 label"""
        img_dir, label = self.item_list[idx]
        # img_name = basename(img_path)   # 返回路径中的最后文件名
        # _, extension = splitext(img_name)    # 获得文件拓展名
        # print(extension)
        # image = self.image_loader(img_path, extension)
        # print(image)
        # if self.transform is not None:
        #     image = self.transform(image)
        # return image, label, img_path

        name_list = listdir(img_dir)
        tensor_list = []
        for item in name_list:
            img_position, extension = splitext(item)
            if int(img_position) in self.select_list:    # 只读入相应位置的图片
                img = Image.open(join(img_dir,item))
                tensor_list.append(self.transform(img))
        result = cat(tensor_list, dim=0)
        return result, label, img_dir
        


    def image_loader(self, img_name, extension):
        """
        输入：img_name：图片具体路径
            extension：拓展名
        输出：img 对象
        作用：根据拓展名读取图片文件，dcm文件格式的图片需单独处理
        """
        if extension == '.JPG':
            # print('读取jpg')
            return self.read_jpg(img_name)
        elif extension == '.jpg':
            # print('读取jpg')
            return self.read_jpg(img_name)
        elif extension == '.DCM':
            # print('读取dcm')
            return self.read_dcm(img_name)
        elif extension == '.dcm':
            # print('读取dcm')
            return self.read_dcm(img_name)
        elif extension == '.Bmp':
            # print('读取Bmp')
            return self.read_bmp(img_name)
        elif extension == '.png':
            return self.read_png(img_name)

    def read_jpg(self, img_name):
        return Image.open(img_name)

    def read_dcm(self, img_name):
        """使用 SITK 医学图像处理接口读取图片，并转化成 img 对象"""
        ds = sitk.ReadImage(img_name)
        img_array = sitk.GetArrayFromImage(ds)
        img_bitmap = Image.fromarray(img_array[0])
        return img_bitmap

    def read_bmp(self, img_name):
        return Image.open(img_name)

    def read_png(self, img_name):
        return Image.open(img_name)

if __name__ == '__main__':

    # 计算三通道图像的 mean 和 std 时使用
    # tf = transforms.Compose([
    #     # transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    # select_list=list(range(0,9))
    # dataset = AltrasoundDataset(root_dir='/media/huangyujun/disk/workspace/Altrasound_pro/seperated_dataset',transform=tf)
    # dataloader = DataLoader(dataset=dataset, batch_size=int(len(dataset)), shuffle=False, num_workers=4)
    # img_l = []
    # for imgs, label, img_dir in dataloader:
    #     imgs = imgs.numpy()
    #     for i in range(len(select_list)):
    #         # 对三通道图像时计算mean和std使用
    #         # img_l.append(imgs[:,i:i+3,:,:])
    #     all = np.vstack(img_l)
    #     print(all.shape)
    #     mean = np.mean(all, axis=(0, 2, 3))
    #     std = np.std(all, axis=(0, 2, 3))
    # print(mean, std)
    
    # 计算灰度图时使用
    tf = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    select_list=list(range(0,9))
    dataset = AltrasoundDataset(root=environ['ALTRASOUND'],transform=tf, dataset_type='all')
    dataloader = DataLoader(dataset=dataset, batch_size=int(len(dataset)), shuffle=False, num_workers=4)
    img_l = []
    for imgs, label, img_dir in dataloader:
        imgs = imgs.numpy()
        print(imgs.shape)
        for i in range(len(select_list)):
            # 对于单通道时计算mean和std使用
            img_l.append(imgs[:,i,:,:])
        all = np.vstack(img_l)
        print(all.shape)
        mean = np.mean(all)
        std = np.std(all)
    print(mean, std)