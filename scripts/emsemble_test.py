from genericpath import exists
from posixpath import dirname
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
from os.path import basename, dirname, join
from shutil import copyfile
from utils.tools import check_mkdir

# 选择模型集成的方式：多数投票 或 输出平均
# mode = 'multi_vote' # 'multi_vote' 或 'output_avg'
mode = 'output_avg'

def copy_error_img(save_basedir, all_path, all_pred, all_label, all_scores):
    save_dir = os.path.join(save_basedir, 'error_img')
    check_mkdir(save_dir)
    img_dir = './重建括约肌'
    error_idxs = torch.where(all_pred != all_label)[0].tolist()
    error_path = [all_path[i] for i in error_idxs]
    error_score = [all_scores[i] for i in error_idxs]
    
    error_csv = open(join(save_dir,'predict_probability.csv'), "w")
    error_writer = csv.writer(error_csv, dialect = "excel")
    error_writer.writerow(['图片名\\类别', 'nonstandard','standard'])
        
    for idx, splited_path in enumerate(error_path):
        class_name = '合格' if basename(dirname(splited_path))=='standard' else '不合格'
        img_name = class_name+basename(splited_path)+'.bmp'
        try:
            copyfile(join(join(img_dir,class_name),img_name), join(save_dir, img_name))
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        error_writer.writerow([class_name+basename(splited_path)]+error_score[idx].numpy().tolist())

    error_csv.close()


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
    
    pblog.debug('models to be used stored at : '+args.model_load_dir)
    pblog.debug('models emsemble mode : '+mode)
    save_dir = os.path.join(args.model_load_dir,'..')
    dict_list = tools.load_model_dict(args.model_load_dir)
    
    # 模型集成列表
    ensamble_models = []
    ensamble_dataloader = []
    # 记录结果
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    all_scores = torch.tensor([])
    path_list = []
    error_list = []
    loss_l = []
    loss_n = []
    # 数据记录文件
    csv_name = '{}_model_error.csv'.format(mode)
    error_csv = open(os.path.join(save_dir,csv_name), "w")
    error_writer = csv.writer(error_csv, dialect = "excel")
    # 将loss、acc 写入表格
    record_dic = {}
    record_dic['name'] = ['test acc', 'test n_precision', 'test n_recall', 'test n_specificity', 'test n_f1',
        'test s_precision', 'test s_recall', 'test s_specificity', 'test s_f1']

    for dict in dict_list:
        model_type, select_list = dict['model_type'], dict['select_list']
        in_channels = len(dict['select_list'])

        img_size = [350,600]
        if dict['model_type'] in ['vit', 'se_resnet', 'swin-transfomer', 'tnt_s']:
            img_size = [224,224]

        # img_size = dict['img_size']

        # 准备处理数据方法
        gray_tf = transforms.Compose([
                    # se_resnet
                    transforms.Resize(img_size),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_gray, std_gray)])
        # 准备数据集
        tf = gray_tf
        test_dataset = AltrasoundDataset(args.Altrasound_data_dir,transform=tf,
            select_list=select_list,dataset_type='test')
        # test_dataset = AltrasoundDataset(args.Altrasound_data_dir,transform=tf,
        #     select_list=select_list,dataset_type='all')

        args.class_names = test_dataset.class_names
        # 仍然使用 batch size 是为了节省显存，shuffle默认为False（保证了输入数据一致）
        dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        # 准备模型
        model = get_net(model_type, test_dataset.class_num, in_channels).cuda()
        model.load_state_dict(dict['model_dic'])
        model.eval()
        ensamble_models.append(model)
        ensamble_dataloader.append(dataloader.__iter__())
        # ensamble_dataloader.append(dataloader)

        pblog.info('Loaded weights for {} : select_list={} , img_size={}'.format(
            dict['model_type'],dict['select_list'],img_size))

    # 自己写的一个迭代方法
    with torch.no_grad():
        while(1):
            try:
                scores = torch.zeros((args.batch_size,2))
                for idx in range(len(ensamble_models)):
                    inputs, labels, image_paths = next(ensamble_dataloader[idx])
                    model = ensamble_models[idx]
                    inputs= inputs.cuda(non_blocking=True)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    if idx==0 : 
                        path_list.extend(image_paths)
                        all_labels = torch.cat((all_labels, labels), 0)
                    
                    if mode == 'multi_vote':
                        # 多数投票
                        preds = preds.detach().cpu().tolist()
                        scores[list(range(len(preds))),preds] = scores[list(range(len(preds))),preds]+1
                    elif mode == 'output_avg':
                        # 求logits平均
                        scores[0:outputs.shape[0]] += torch.softmax(outputs.detach().cpu(), 1)
                    else:
                        pblog.warning('unknown mode!')
                
                _, preds = torch.max(scores,1)
                all_preds = torch.cat((all_preds, preds), 0)
                all_scores = torch.cat((all_scores, scores), 0)                
            except StopIteration:
                pblog.debug('test set inference and evaluation finished')
                break

    # 不足batch size 的情况
    minus = labels.shape[0] - args.batch_size
    all_preds = all_preds[0:minus]
    all_scores = all_scores[0:minus]

    # 将 confusion matrix 和 ROC curve 输出到 图片文件
    cm_name = "Emsemble models' Confusion Matrix"
    cm_figure, tp, fp, fn, tn = tools.plot_confusion_matrix(all_labels, all_preds, args.class_names, cm_name)
    cm_figure.savefig(os.path.join(save_dir,'{}_emsemble_cm.png'.format(mode)))
    if mode == 'output_avg':
        roc_name = "Emsemble models' ROC Curve"
        all_scores /= len(ensamble_models)*1.0
        roc_auc, figure, opt_threshold, opt_point = tools.plot_ROC_curve(all_labels, all_scores, args.class_num, roc_name)
        figure.savefig(os.path.join(save_dir,'{}_emsemble_ROC.png'.format(mode)))

    # 计算 precision 和 recall， 将 zero_division 置为0，使当 precision 为0时不出现warning
    acc = torch.sum(all_preds == all_labels).item() / len(all_labels)
    # precision = precision_score(all_labels, all_preds, pos_label=0, zero_division=0)
    # nonstandard 的 precision, recall specificity
    indicator_n = {}
    indicator_s = {}
    indicator_n['precision'] = tp / (tp + fp)
    indicator_n['recall'] = tp / (tp + fn)
    indicator_n['specificity'] = tn / (tn + fp)
    # standard 的 precision, recall, specificity
    indicator_s['precision'] = tn / (tn + fn)
    indicator_s['recall'] = tn / (tn + fp)
    indicator_s['specificity'] = tp / (tp + fn)

    if acc == 0. and indicator_n['recall'] == 0.:
        indicator_n['f1_score'] = float('nan')
    else:
        indicator_n['f1_score'] = 2 * (indicator_n['precision'] * indicator_n['recall']) / (indicator_n['precision'] + indicator_n['recall'])

    if acc == 0. and indicator_s['recall'] == 0.:
        indicator_s['f1_score'] = float('nan')
    else:
        indicator_s['f1_score'] = 2 * (indicator_s['precision'] * indicator_s['recall']) / (indicator_s['precision'] + indicator_s['recall'])

    pblog.debug('Acc = {:0<5.5f}'.format(acc))
    pblog.debug('Nonstandard Precision = {:0<5.5f} , Nonstandard Recall = {:0<5.5f} , Nonstandard Specificity={:0<5.5f} , Nonstandard F1 = {:0<5.5f}'
        .format(indicator_n['precision'], indicator_n['recall'], indicator_n['specificity'], indicator_n['f1_score']))
    pblog.debug('Standard Precision = {:0<5.5f} , Standard Recall = {:0<5.5f} , Standard Specificity={:0<5.5f} , Standard F1 = {:0<5.5f}'
        .format(indicator_s['precision'], indicator_s['recall'], indicator_s['specificity'], indicator_s['f1_score']))
    
    pblog.debug('=' * 20)
    pblog.info(' ')

    record_dic['emsemble model'] = [acc, indicator_n['precision'], indicator_n['recall'], 
        indicator_n['specificity'],indicator_n['f1_score'], 
        indicator_s['precision'], indicator_s['recall'], indicator_s['specificity'], indicator_s['f1_score']]

    # 找出分类错误的图片的路径，以 图片路径+真实类标号 形式存储
    for index in torch.where(all_preds != all_labels)[0].tolist():
        error_list.append((path_list[index],all_labels[index]))

    error_writer.writerow(['emsemble model'])
    error_writer.writerow(['Image Name', 'real label', 'path'])
    for img_path, real_label in error_list:
        if real_label.item() == 0:
            error_writer.writerow([os.path.basename(img_path), 'nonstandard', img_path])
        else:
            error_writer.writerow([os.path.basename(img_path), 'standard', img_path])
    error_writer.writerow([' '])
    
    csv_name = '{}_emsemble_evaluate.csv'.format(mode)
    pd.DataFrame(record_dic).to_csv(os.path.join(save_dir,csv_name))
    error_csv.close()

    data_df = pd.DataFrame(all_scores.numpy())
    data_df.index=path_list
    data_df.columns = ['nontandard', 'standard']
    data_df['real label']=all_labels.numpy()
    data_df['predict']=all_preds.numpy()
    data_df.to_csv(os.path.join(save_dir,'{}_predict.csv'.format(mode)))

    # copy_error_img(save_dir, path_list, all_preds, all_labels, all_scores)

