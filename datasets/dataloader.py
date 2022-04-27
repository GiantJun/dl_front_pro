# @ FileName: dataset.py
# @ Author: Yujun
# @ Time: 21-8-3 下午9:17
import os
from os.path import join
from PIL import Image
import numpy as np
import torch
import torchvision
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from scripts import gallbladder_dataset
from torchvision import datasets
from scripts import altrasound_dataset


class BaseLoader:
    def __init__(self, args):
        # special params
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size

        # custom properties
        self._dataset_data = None
        self._dataset_train = None
        self._dataset_eval = None
        self._dataset_test = None
        self._dataset_norm = None
        self._dataset_attack = None
        self.dataloader_train = None
        self.dataloader_eval = None
        self.dataloader_test = None

    @property
    def dataset_data(self):
        raise NotImplementedError

    @property
    def dataset_train(self):
        raise NotImplementedError

    @property
    def dataset_eval(self):
        raise NotImplementedError

    @property
    def dataset_test(self):
        raise NotImplementedError

    @property
    def dataset_norm(self):
        raise NotImplementedError

    @property
    def data(self):
        if self.dataloader_data is None:   #注意：此处的 self.dataset_train 经过方法重载，具有个性化功能
            self.dataloader_data = Data.DataLoader(self.dataset_data,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=self.num_workers,
                                                    pin_memory=True)
        return self.dataloader_data

    @property
    def train(self):
        if self.dataloader_train is None:   #注意：此处的 self.dataset_train 经过方法重载，具有个性化功能
            self.dataloader_train = Data.DataLoader(self.dataset_train,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=self.num_workers,
                                                    pin_memory=True)
        return self.dataloader_train

    @property
    def eval(self):
        if self.dataloader_eval is None:    #注意：此处的 self.dataset_eval 经过方法重载，具有个性化功能
            self.dataloader_eval = Data.DataLoader(self.dataset_eval,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.num_workers,
                                                   pin_memory=True)
        return self.dataloader_eval

    @property
    def test(self):
        if self.dataloader_test is None:    #注意：此处的 self.dataset_test 未经过重载
            self.dataloader_test = Data.DataLoader(self.dataset_test,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.num_workers,
                                                   pin_memory=True)
        return self.dataloader_test

    def cal_norm(self, n=1):
        return None

    @staticmethod
    def random_sample_base(base_dir, transform, size):
        classes = os.listdir(base_dir)
        classes.sort()
        images = []
        for c in classes:
            folder = join(base_dir, c)
            for file_name in np.random.choice(os.listdir(folder), size, False):
                img = join(folder, file_name)
                images.append(transform(Image.open(img)))
        return classes, torch.stack(images)


class CIFAR10(BaseLoader):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    num_classes = 10
    class_names = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, args):
        super(CIFAR10, self).__init__(args)
        self.base_dir = args.CIFAR10_dir
        self.mean = CIFAR10.mean
        self.std = CIFAR10.std
        self.class_names = CIFAR10.class_names

    @property
    def dataset_train(self):
        if self._dataset_train is None:
            tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])
            self._dataset_train = torchvision.datasets.CIFAR10(
                root=self.base_dir, train=True, transform=tf, download=True)
        return self._dataset_train

    @property
    def dataset_eval(self):
        if self._dataset_eval is None:
            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])
            self._dataset_eval = torchvision.datasets.CIFAR10(
                root=self.base_dir, train=False, transform=tf)
        return self._dataset_eval

    @property
    def dataset_norm(self):
        if self._dataset_norm is None:
            self._dataset_norm = torchvision.datasets.CIFAR10(
                self.base_dir, transform=transforms.ToTensor())
        return self._dataset_norm


class ImageNet100(BaseLoader):
    mean = [0.47881872, 0.45927624, 0.41515172]
    std = [0.27191086, 0.26549916, 0.27758414]
    num_classes = 100
    class_names = [str(i) for i in range(num_classes)]

    def __init__(self, args):
        super(ImageNet100, self).__init__(args)
        # base image dir
        self.base_dir = args.ImageNet100_dir
        self.mean = ImageNet100.mean
        self.std = ImageNet100.std
        self.img_size = args.img_size

    @property
    def dataset_train(self):
        if self._dataset_train is None:
            tf = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
                transforms.RandomRotation([-180, 180]),
                transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                                        scale=[0.7, 1.3]),
                transforms.RandomCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            self._dataset_train = torchvision.datasets.ImageFolder(
                join(self.base_dir, 'train'), transform=tf)
        return self._dataset_train

    @property
    def dataset_eval(self):
        if self._dataset_eval is None:
            tf = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            self._dataset_eval = torchvision.datasets.ImageFolder(
                join(self.base_dir, 'val'), transform=tf)
        return self._dataset_eval

    @property
    def dataset_norm(self):
        if self._dataset_norm is None:
            tf = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomCrop(self.img_size),
                transforms.ToTensor()
            ])
            self._dataset_norm = ImageFolder(join(self.base_dir, 'train'),
                                             transform=tf)
        return self._dataset_norm

    def cal_norm(self, n=10):
        ds = self.dataset_norm
        dl = Data.DataLoader(dataset=ds, batch_size=self.batch_size,
                             num_workers=self.num_workers, pin_memory=True)
        m1 = m2 = m3 = s1 = s2 = s3 = 0
        for i in range(n):
            print('times:{}'.format(i))
            ll = len(dl)
            for idx, (x, _) in enumerate(dl):
                print('iter {} of {}'.format(idx, ll))
                x = x.cuda(non_blocking=True)
                m1 += x[:, 0, :, :].mean().item()
                m2 += x[:, 1, :, :].mean().item()
                m3 += x[:, 2, :, :].mean().item()
                s1 += x[:, 0, :, :].std().item()
                s2 += x[:, 1, :, :].std().item()
                s3 += x[:, 2, :, :].std().item()

        n = n * len(dl)
        print('mean: ', m1 / n, m2 / n, m3 / n)
        print('std: ', s1 / n, s2 / n, s3 / n)

    def random_sample(self, size):
        base_dir = join(self.base_dir, 'eval')
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor()
        ])
        return self.random_sample_base(base_dir, transform, size)


def get_dataloader(args):
    """
    输入：args：参数字典，主要使用到‘dataset’这个参数，用于指定数据集
    输出：数据集对应的dataloader对象
    作用：根据 args['dataset'] 选择生成对象
    """
    if args.dataset == 'ImageNet100':
        return ImageNet100(args)
    if args.dataset == 'CIFAR10':
        return CIFAR10(args)
    if args.dataset == 'Gallbladder':
        return Gallbladder(args)
    if args.dataset == 'Altrasound':
        return Altrasound(args)
    else:
        raise ValueError('No dataset: {}'.format(args.dataset))

class Altrasound(BaseLoader):
    """自定义 dataloader"""
    # contain empty
    # transforms 进行 normalization 需要的参数
    class_names = ['nonstandard', 'standard']
    num_classes = len(class_names)

    def __init__(self, args):
        super(Altrasound, self).__init__(args)
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        # mean = [0.359, 0.361, 0.379]
        # std = [0.190, 0.190, 0.199]
        # except empty
        # mean = [0.359, 0.361, 0.380]
        # std = [0.191, 0.190, 0.200]

        self.img_size = args.img_size
        # 将图片转换为灰度图，3代表输入图片的通道数
        self.convert = transforms.Grayscale(3)

    @property
    def dataset_train(self):
        """
        输入：无
        输出：自定义训练数据集对象
        作用：构造自定义在 training 子数据集上的 dataset 对象，读入数据并做变换，为属性方法，可不加括号调用
        """
        if self._dataset_train is None:
            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            self._dataset_train = datasets.ImageFolder('/media/huangyujun/disk/workspace/Altrasound_Pro/dataset_merged/train',transform=tf)
        return self._dataset_train

    @property
    def dataset_eval(self):
        """
        输入：无
        输出：自定义训练数据集对象
        作用：构造自定义在 test 子数据集上的 dataset 对象，读入数据并做变换，为属性方法，可不加括号调用
        """
        if self._dataset_eval is None:
            tf = transforms.Compose([
                self.gray_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            self._dataset_eval = datasets.ImageFolder('/media/huangyujun/disk/workspace/Altrasound_Pro/dataset_merged/valid',transform=tf)
        return self._dataset_eval

    @property
    def dataset_data(self):
        """
        输入：无
        输出：自定义训练数据集对象
        作用：构造自定义在 test 子数据集上的 dataset 对象，读入数据并做变换，为属性方法，可不加括号调用
        """
        if self._dataset_data is None:
            self._dataset_data = altrasound_dataset('/media/huangyujun/disk/workspace/Altrasound_Pro/datasets')
        return self._dataset_data    

    def set_data_transform(self, tf):
        """
        输入：tf：需要设置的 transform 对象
        输出： 无
        作用：在使用 dataloader 迭代获取元素前，为没有 transform 的 dataset_data 做初始化
        """
        self._dataset_data.set_transform(tf)

    def gray_to_rgb(self, img):
        """
        输入：img：灰度图片对象
        输出：转化为彩色图的对象
        """
        # 发现代码有错误，按照下面的运行
        channels = len(img.split())
        if channels < 3:
            img = self.convert(img)
        return img

class Gallbladder(BaseLoader):
    """自定义 dataloader"""
    # contain empty
    # transforms 进行 normalization 需要的参数
    mean = [0.359, 0.361, 0.379]
    std = [0.190, 0.190, 0.199]
    # except empty
    # mean = [0.359, 0.361, 0.380]
    # std = [0.191, 0.190, 0.200]

    num_classes = 2
    class_names = ('Biliary atresia', 'Non-biliary atresia')

    def __init__(self, args):
        super(Gallbladder, self).__init__(args)
        self.train_dir = args.Gallbladder_train_dir
        self.test_dir = args.Gallbladder_test_dir
        self.train_csv = args.train_csv
        self.test_csv = args.test_csv
        self.mean = Gallbladder.mean
        self.std = Gallbladder.std
        self.img_size = args.img_size
        # 将图片转换为灰度图，3代表输入图片的通道数
        self.convert = transforms.Grayscale(3)

    @property
    def dataset_train(self):
        """
        输入：无
        输出：自定义训练数据集对象
        作用：构造自定义在 training 子数据集上的 dataset 对象，读入数据并做变换，为属性方法，可不加括号调用
        """
        if self._dataset_train is None:
            tf = transforms.Compose([
                # transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
                # transforms.RandomRotation([-180, 180]),
                # transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                #                         scale=[0.7, 1.3]),
                # transforms.RandomCrop(self.img_size),
                self.gray_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            self._dataset_train = gallbladder_dataset.GallbladderDataset(csv_file=join(self.train_dir, self.train_csv),
                                                                         root_dir=self.train_dir,
                                                                         transform=tf)
        return self._dataset_train

    @property
    def dataset_eval(self):
        """
        输入：无
        输出：自定义训练数据集对象
        作用：构造自定义在 test 子数据集上的 dataset 对象，读入数据并做变换，为属性方法，可不加括号调用
        """
        if self._dataset_eval is None:
            tf = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                self.gray_to_rgb,
                # transforms.RandomCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            self._dataset_eval = gallbladder_dataset.GallbladderDataset(csv_file=join(self.test_dir, self.test_csv),
                                                                        root_dir=self.test_dir,
                                                                        transform=tf)
        return self._dataset_eval

    def gray_to_rgb(self, img):
        """
        输入：img：灰度图片对象
        输出：转化为彩色图的对象
        """
        # # 以矩阵的方式载入，做转置，获取图片尺寸
        # size = np.array(img).transpose().shape
        # print(size)
        # # 通道数不等于3
        # if size[0] != 3:
        #     # 此处将调用 convert 成员变量，即一个transforms.Grayscale，指定了输出图片的通道数
        #     img = self.convert(img)

        # 发现代码有错误，按照下面的运行
        channels = len(img.split())
        if channels !=3:
            img = self.convert(img)
        return img


