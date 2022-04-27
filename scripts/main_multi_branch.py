#coding:utf-8
# @ FileName: main.py
# @ Author: Alexis
# @ Time: 20-11-28 下午9:17

from ast import arg
from utils.settings import settings
from utils.logger import get_logger
from scripts.trainer import Trainer
from networks.net import get_net
from datasets.dataset import AltrasoundDataset, mean_RGB, std_RGB, mean_gray, std_gray
from datasets import myTransform
import torchvision.transforms as transforms
from os.path import join
import torch
import numpy as np
import random
import os
import csv

from utils import tools

random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
random.seed(random_seed)
np.random.seed(random_seed)

is_fix = False
is_concat = True

if __name__ == '__main__':
    # 产生程序运行所需要的参数,有两种方式产生，1默认值：在tool.py中实现，2由bash传入，3根据本文件当前地址生成
    # 在config中使用了global方法，将变量值存储在静态全局
    args = settings()  # __file__当前文件的绝对路径
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # 产生一个logger，用于输出信息
    logger = get_logger(args, True)
    pblog = logger['pblog']
    tblog = logger['tblog']
    try:
        pblog.debug('###start###')
        if args.cmd == 'analyze': 
            pass
                            
        elif args.cmd == 'train':
            pblog.debug('output file saved at %s' % args.output_dir)
            pblog.debug('training model : %s' % args.model_type)
            pblog.debug('select list : {}'.format(args.select_list))
            pblog.debug('input channels : {}'.format(args.in_channels))
            pblog.debug('input image size : {}'.format(args.img_size))
            pblog.debug('epoch : {}'.format(args.epoch))
            pblog.debug('lr : {}'.format(args.lr))
            pblog.debug('lr_epoch : {}'.format(args.lr_epoch))
            pblog.debug('batch size : {}'.format(args.batch_size))
            pblog.debug('optimizer : {}'.format(args.optimizer))
            pblog.debug('is concat : {}'.format(is_concat))
            pblog.debug('is fix : {}'.format(is_fix))
            pblog.debug('model dir: {}'.format(args.model_load_dir))

            ############### 会自动根据模型类别选择使用 350x600 还是 224x224 对图像进行reshape ###################
            # 训练数据集对象
            tfs = {'train':transforms.Compose([
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(5),
                    myTransform.AddPepperNoise(0.95, p=0.5),
                    transforms.Resize(args.img_size),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_gray, std_gray)]),  # 此处的 mean 和 std 由
                'test':transforms.Compose([
                    transforms.Resize(args.img_size),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_gray, std_gray)])
            }
            model_list = []
            # 创建trainer对象，训练模型
            trainer = Trainer(args, logger)
            if args.model_load_dir is None:
                for i in args.select_list:
                    sub_pic = [i]
                    train_dataset = AltrasoundDataset(args.Altrasound_data_dir,transform=tfs['train'],select_list=sub_pic)  # 0--nonstandard, 1--standard
                    valid_dataset = AltrasoundDataset(args.Altrasound_data_dir,transform=tfs['test'],select_list=sub_pic)  # 0--nonstandard, 1--standard
                    test_dataset = AltrasoundDataset(args.Altrasound_data_dir,transform=tfs['test'],select_list=sub_pic, dataset_type='test')  # 0--nonstandard, 1--standard
                    dataset = {'train':train_dataset, 'valid':valid_dataset, 'test':test_dataset}
                    args.class_num = train_dataset.class_num
                    args.class_names = train_dataset.class_names
            
                    # 创建与训练模型
                    model = get_net(args.model_type, args.class_num, 1).cuda()          
                    
                    pblog.debug('='*10+' Start Single Train , using sub-pic {}'.format(i)+'='*10)
                    save_name = args.desc+'_singel_{}'.format(i)
                    trainer.single_train(args, dataset, model, random_seed, save_name)
                    model_list.append(model.features)
            else:
                dict_list = tools.load_model_dict(args.model_load_dir)
                for i in args.select_list:
                    item = dict_list[i]
                    model = get_net(item['model_type'], args.class_num, 1)
                    print('Loaded {}: select list={} , class_num={}'.format(item['model_type'],item['select_list'],args.class_num))
                    model.load_state_dict(item['model_dic'])
                    model_list.append(model.features)
            
            train_dataset = AltrasoundDataset(args.Altrasound_data_dir,transform=tfs['train'],select_list=args.select_list)  # 0--nonstandard, 1--standard
            valid_dataset = AltrasoundDataset(args.Altrasound_data_dir,transform=tfs['test'],select_list=args.select_list)  # 0--nonstandard, 1--standard
            test_dataset = AltrasoundDataset(args.Altrasound_data_dir,transform=tfs['test'],select_list=args.select_list, dataset_type='test')  # 0--nonstandard, 1--standard
            dataset = {'train':train_dataset, 'valid':valid_dataset, 'test':test_dataset}
            args.class_num = train_dataset.class_num
            args.class_names = train_dataset.class_names 
            model = get_net('multi_branch',args.class_num, model_list=model_list, is_fix=is_fix, is_concat=is_concat).cuda()
            pblog.debug('='*10+' Start Weighted Train '+'='*10)
            save_name = 'multi_branch_concat-{}_fix-{}'.format(is_concat,is_fix)
            trainer.single_train(args, dataset, model, random_seed, save_name)
            
        else:
            raise ValueError('No cmd: {}'.format(args.cmd))
    except:
        pblog.exception('Exception Logged')
        pblog.warning('-'*50)
        exit(1)
    else:
        pblog.info('### finished ###')
        
