import argparse
from os.path import join
from utils import tools
from os import environ


class settings:
    def __init__(self):
        # 项目的基本信息
        AUTHOR = "Yujun"
        PROGRAM = "Ultrasound_Pro"
        DESCRIPTION = "Biliary atresia diagnose. " \
                    "If you find any bug, please new issue. "
        
        cmd_list = ['train', 'test', 'analyze']
        optimizer_list = ['adam', 'sgd']
        scheduler_list = ['steplr', 'coslr', 'explr']
        model_type_list = ['res18', 'res50', 'res101', 'res152', 'Inc-v4', 'IncRes-v2', 
            'pnasnet', 'se_resnet', 'densenet', 'efficientnet', 'vit', 'vgg19', 'swin_transformer', 
            'tnt_b', 'tnt_s']


        parser = argparse.ArgumentParser(prog=PROGRAM, description=DESCRIPTION)
        parser.add_argument('cmd', choices=cmd_list, type=str, default='train', help='what to do')
        parser.add_argument("--local_rank", default=-1)

        # 数据集和使用的模型
        parser.add_argument('--dataset', type=str, default='Altrasound',
                            help='dataset you gonna use')
        # parser.add_argument('--img_size', type=int, default=224, help='size of the input image')
        parser.add_argument('--model_type', choices=model_type_list, type=str, default=None, help='model to use')

        # 设备信息设置
        parser.add_argument('--cuda', type=str, default='0', help='gpu(s)')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='num workers for dataloader')

        # 训练参数设置
        # 几个选项
        parser.add_argument('--optimizer', choices=optimizer_list, type=str, default='sgd',
                            help='optimizer to use')
        parser.add_argument('--scheduler', choices=scheduler_list, type=str, default='steplr',
                            help='sheduler to use')
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
        parser.add_argument('--lr_epoch', type=int, default=35,
                            help='learning rate decay epoch')
        parser.add_argument('--epoch', type=int, default=200, help='epoch')
        parser.add_argument('--pre_train', action='store_true', default=True,
                            help='pre train')
        parser.add_argument('--batch_size', type=int, default=64,
                            help='batch for all')
        # 输出信息设置        
        parser.add_argument('--save_dir_name', type=str, default=None,
                            help='saving path for output files')
        parser.add_argument('--log_name', type=str, default='TestLog', help='log name for logger')
        parser.add_argument('--input_base_dir', type=str, help='input image dirs for grad-cam') # 对应切分好的含9张图片的目录

        parser.add_argument('--kfold', type=int, default=0, help='fold num for kfold validation')
        parser.add_argument('--show_log', action='store_true', default=False, help='print log to the terminal')
        parser.add_argument('--sgd_m', type=float, default=0.9, help='momentun of sgd')
        parser.add_argument('--sgd_w', type=float, default=5e-4, help='weight decay of sgd')
        parser.add_argument('--model_dir', type=str, default=None, help='directory to save model weights')
        parser.add_argument('--model_load_dir', type=str, default=None, help='directory to load model weights')
        parser.add_argument('--grad_cam_dir', type=str, default=None, help='directory to save grad cam image')

        for name, value in vars(parser.parse_args()).items():
            setattr(self, name, value)
        
        self.class_num = 2
        self.class_names = []

        # 各常用数据集的位置
        self.CIFER10_dir = '/media/huangyujun/disk/data/CIFER-10/data' 
        self.ImageNet100_dir = '/media/huangyujun/disk/data/Imagenet2012'

        # go to ~/.bashrc and add envrionment varieble to the end, likes "export ALTRASOUND='/(your data path)' "
        self.train_csv_dir = environ["DLDATASET_TRAIN"]
        self.test_csv_dir = environ["DLDATASET_TEST"]

        self.in_channels = 3
        # 数据集进行归一化、标准化的参数，以及输出类别数量
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5,0.5,0.5]

        # if self.model_type in ['se_resnet', 'vit', 'swin_transformer', 'tnt_b', 'tnt_s']:
        #     self.img_size = [224,224]
        # else:
        #     self.img_size = [350,600]
        
        self.img_size = [224,224]

        # 创建输出文件夹
        if self.cmd == 'train':
            if self.save_dir_name is None:
                self.save_dir_name = tools.gen_name(self.model_type,'')
            self.sub_dir = join('./output/', self.save_dir_name)

            if tools.check_mkdir(self.sub_dir) is True:
                print('='*10+'{} already existed , program will auto add suffix.'.format(self.sub_dir)+'='*10)
                self.sub_dir = join('./output/', tools.gen_name(self.save_dir_name,''))

            self.log_dir = join(self.sub_dir, 'log')
            self.tb_dir = join(self.sub_dir, 'tb')
            if self.model_dir == None:
                self.model_dir = join(self.sub_dir, 'model')
            if self.grad_cam_dir == None:
                self.grad_cam_dir = join(self.sub_dir, 'grad_cam_image')

            self.desc = '{}_{}_{}'.format(self.dataset, self.model_type, self.optimizer)
            
            tools.check_mkdir(self.log_dir)
            tools.check_mkdir(self.tb_dir)
            tools.check_mkdir(self.model_dir)
            tools.check_mkdir(self.grad_cam_dir)

    
    

