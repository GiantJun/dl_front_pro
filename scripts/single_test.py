import numpy as np
import torch
from torch.serialization import save
from utils.settings import settings
import os
from utils.logger import get_logger
from utils import tools
from networks.net import get_net
import torchvision.transforms as transforms
from datasets.dataset import AltrasoundDataset, mean_RGB, std_RGB, mean_gray, std_gray
from torch.utils.data import DataLoader
import csv
import pandas as pd
from scripts.trainer import Trainer

if __name__ == '__main__':
    args = settings()  # __file__当前文件的绝对路径
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # 产生一个logger，用于输出信息
    logger = get_logger(args, True)
    pblog = logger['pblog']
    tblog = logger['tblog']

    # if args.cmd is not 'test':
    #     pblog.exception('cmd is not test')
    #     pblog.warning('-'*50)
    #     exit(1)
    # 创建agent对象，训练模型
    agent = Trainer(args, logger)
    
    pblog.debug('models to be used stored at : '+args.model_load_dir)
    save_dir = os.path.join(args.model_load_dir,'..')
    dict_list = tools.load_model_dict(args.model_load_dir)
    
    # 模型集成列表
    ensamble_models = []
    ensamble_dataloader = []
    # 记录结果
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    # all_scores = torch.tensor([])
    path_list = []
    error_list = []
    loss_l = []
    loss_n = []
    # # 数据记录文件
    # csv_name = 'kfold_model_error.csv'
    # error_csv = open(os.path.join(save_dir,csv_name), "w")
    # error_writer = csv.writer(error_csv, dialect = "excel")
    # 将loss、acc 写入表格
    record_dic = {}

    record_dic['name'] = ['test acc', 'test auc', 'test n_precision', 'test n_recall', 'test n_specificity', 'test n_f1',
        'test s_precision', 'test s_recall', 'test s_specificity', 'test s_f1']

    predic_dic = {}

    for idx, dict in enumerate(dict_list):
        model_type, select_list = dict['model_type'], dict['select_list']
        in_channels = len(dict['select_list'])
        args.model_type = model_type

        img_size = [350,600]
        if dict['model_type'] in ['vit', 'se_resnet', 'swin-transformer', 'tnt_s']:
            img_size = [224,224]

        # 准备处理数据方法
        gray_tf = transforms.Compose([
                    # se_resnet
                    transforms.Resize(img_size),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_gray, std_gray)])

        # 准备数据集
        tf = gray_tf # if dict['is_input_gray'] else rgb_tf
        test_dataset = AltrasoundDataset(args.Altrasound_data_dir,transform=tf,
            select_list=select_list,dataset_type='test')
        args.class_names = test_dataset.class_names
        # 仍然使用 batch size 是为了节省显存，shuffle默认为False（保证了输入数据一致）
        dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        # 准备模型
        model = get_net(model_type, test_dataset.class_num, in_channels).cuda()
        model.load_state_dict(dict['model_dic'])
        model.eval()
        # 评价模型
        pblog.info('Evaluating model for {} in fold {} , select_list={} , is_input_gray=True'.format(
            dict['model_type'], idx, dict['select_list']))
        acc, roc_auc, indicator_n, indicator_s, error_list, pre_list, all_scores, label_list, path_list = agent._evaluate_model(model, dataloader, idx)
        
    
        record_dic[str(idx)] = [acc, roc_auc, indicator_n['precision'], indicator_n['recall'],
            indicator_n['specificity'],indicator_n['f1_score'], 
            indicator_s['precision'], indicator_s['recall'], indicator_s['specificity'], indicator_s['f1_score']]

        # 找出分类错误的图片的路径，以 图片路径+真实类标号 形式存储
        for index in torch.where(all_preds != all_labels)[0].tolist():
            error_list.append((path_list[index],all_labels[index]))

        if idx == 0:
            predic_dic['path'] = path_list
            predic_dic['real label'] = label_list
        predic_dic['predict'+str(idx)]=pre_list
        data_df = pd.DataFrame(all_scores.numpy())
        data_df.index=path_list
        data_df.columns = ['standard', 'nonstandard']
        data_df['real label']=label_list
        data_df['predict']=pre_list
        data_df.to_csv(os.path.join(save_dir,'model_'+str(idx)+'_predict.csv'))
        # predic_dic['score'+str(idx)]=

        # error_writer.writerow(['fold {} model'.format(idx)])
        # error_writer.writerow(['Image Name', 'real label', 'path'])
        # for img_path, real_label in error_list:
        #     if real_label.item() == 0:
        #         error_writer.writerow([os.path.basename(img_path), 'nonstandard', img_path])
        #     else:
        #         error_writer.writerow([os.path.basename(img_path), 'standard', img_path])
        # error_writer.writerow([' '])
        # error_list = []

        # cm_name = "model {} Confusion Matrix".format(idx)
        # cm_figure, tp, fp, fn, tn = tools.plot_confusion_matrix(all_labels, all_preds, args.class_names, cm_name)
        # cm_figure.savefig(os.path.join(save_dir,'model{}_test_cm.png'.format(idx)))
    
    csv_name = 'kfold_test_evaluate.csv'
    pd.DataFrame(record_dic).to_csv(os.path.join(save_dir,csv_name))
    csv_name = 'kfold_test_predict.csv'
    pd.DataFrame(predic_dic).to_csv(os.path.join(save_dir,csv_name))
    # error_csv.close()