#coding:utf-8
# @ FileName: main.py
# @ Author: Alexis
# @ Time: 20-11-28 下午9:17

from utils.settings import settings
from utils.logger import get_logger
from scripts.trainer import Trainer
from networks.net import get_net
from datasets import myTransform
import torchvision.transforms as transforms
from os.path import join
import torch
import numpy as np
import random
import os
import csv
from torch.nn import DataParallel
from datasets.dl_dataset import DLDataset

from utils import tools

random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
random.seed(random_seed)
np.random.seed(random_seed)

if __name__ == '__main__':
    # 产生程序运行所需要的参数,有两种方式产生，1默认值：在tool.py中实现，2由bash传入，3根据本文件当前地址生成
    # 在config中使用了global方法，将变量值存储在静态全局
    args = settings()  # __file__当前文件的绝对路径
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device_ids = range(torch.cuda.device_count())
    # 产生一个logger，用于输出信息
    logger = get_logger(args, True)
    pblog = logger['pblog']
    tblog = logger['tblog']
    try:
        pblog.debug('###start###')
        if args.cmd == 'analyze': 
            pblog.debug('models to be used stored at : '+args.model_load_dir)
            pblog.debug('applying model : '+ args.model_load_dir)
            pblog.debug('grad_cam image saved at : '+args.grad_cam_dir)
            dict_list = tools.load_model_dict(args.model_load_dir)
            args.in_channels = 3
            model = get_net(args.model_type, args.class_num, args.in_channels).cuda()
            model = DataParallel(model)
            
            # 处理输入图片目录
            input_image_dirs = []
            for item in os.listdir(args.input_base_dir):
                class_dir = os.path.join(args.input_base_dir, item) # 得到类似 seperate_dataset/nonstandard 形式的目录
                for image_no in os.listdir(class_dir):
                    input_image_dirs.append(os.path.join(class_dir,image_no))

            for idx, dict in enumerate(dict_list):
                model.load_state_dict(dict['model_dic'])
                for input_image_dir in input_image_dirs:
                    image_no = os.path.basename(input_image_dir)
                    image_class = os.path.basename(os.path.dirname(input_image_dir))
                    save_dir = os.path.join(os.path.join(args.grad_cam_dir,image_class),image_no)
                    tools.check_mkdir(save_dir)
                    cam_img = tools.gen_grad_cam(model, input_image_dir, args.select_list, save_dir, idx, dict_list[0]['is_input_gray'])
                    pblog.info('处理完成 model{} : {}'.format(idx, input_image_dir))
                pblog.info('='*30)
                            
        elif args.cmd == 'train':
            pblog.debug('using {} GPUs , cuda = {}'.format(torch.cuda.device_count(), args.cuda))
            pblog.debug('output file saved at %s' % args.sub_dir)
            pblog.debug('training model : %s' % args.model_type)
            pblog.debug('input channels : {}'.format(args.in_channels))
            pblog.debug('input image size : {}'.format(args.img_size))
            pblog.debug('kfold num : {}'.format(args.kfold))
            pblog.debug('epoch : {}'.format(args.epoch))
            pblog.debug('lr : {}'.format(args.lr))
            pblog.debug('lr_epoch : {}'.format(args.lr_epoch))
            pblog.debug('batch size : {}'.format(args.batch_size))
            pblog.debug('optimizer : {}'.format(args.optimizer))
            pblog.debug('scheduler : {}'.format(args.scheduler))

            ############### 会自动根据模型类别选择使用 350x600 还是 224x224 对图像进行reshape ###################
            # 训练数据集对象
            tfs = {'train':transforms.Compose([
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(20),
                    myTransform.AddPepperNoise(0.95, p=0.5),
                    transforms.Resize(args.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(0.36541346, 0.1908806)]),  # 此处的 mean 和 std 由
                'test':transforms.Compose([
                    transforms.Resize(args.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(0.36541346, 0.1908806)])
            }
            train_dataset = DLDataset(args.train_csv_dir,transform=tfs['train'])  # 0--nonstandard, 1--standard
            valid_dataset = DLDataset(args.train_csv_dir,transform=tfs['test'])  # 0--nonstandard, 1--standard
            test_dataset = DLDataset(args.test_csv_dir,transform=tfs['test'])  # 0--nonstandard, 1--standard
            dataset = {'train':train_dataset, 'valid':valid_dataset, 'test':test_dataset}
            args.class_num = train_dataset.class_num
            args.class_names = train_dataset.class_names
           
            # 创建与训练模型
            model = get_net(args.model_type, args.class_num, args.in_channels).cuda()
            if len(device_ids)>1:
                model=DataParallel(model)       
            
            # 创建agent对象，训练模型
            agent = Trainer(args, logger)
            if args.kfold > 1:
                pblog.debug('='*10+' Start K-Fold Train '+'='*10)
                agent.kfold_train(args, dataset, model, random_seed)
            elif args.kfold ==1:
                pblog.debug('='*10+' Start Single Train '+'='*10)
                save_name = args.desc+'_singel'
                agent.single_train(args, dataset, model, random_seed, save_name)          
        else:
            raise ValueError('No cmd: {}'.format(args.cmd))
    except:
        pblog.exception('Exception Logged')
        pblog.warning('-'*50)
        exit(1)
    else:
        pblog.info('### finished ###')
        
