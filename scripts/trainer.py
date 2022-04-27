# @ FileName: trainer.py
# @ Author: Yujun
# @ Time: 21-8-3 上午9:17
import logging
from asyncio.log import logger
import os
from os.path import join, basename, dirname
import numpy as np
import torch
import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
import torch.nn as nn
from torch.nn.functional import softmax
import copy
# from sklearn.metrics import precision_score, recall_score
import time
from tqdm import tqdm
from utils import tools
from torch.utils.data import SubsetRandomSampler, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import csv

class Trainer:

    def __init__(self, args, logger):
        
        """输入：args，参数字典
        设置各类参数信息，创建模型，创建logger"""
        # 保存参数列表
        self.args = args
        # 设置使用的显卡编号
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss()

        # logger setup
        self.pblog = logger['pblog']
        # 生成一个 SummaryWriter ，用于tensorboard可视化网络结构和结果
        self.tblog = logger['tblog']

    def __del__(self):
        if hasattr(self, 'tblog') and self.tblog is not None:
            self.tblog.close()

    def _train(self, model, dataloaders, fold=''):
        # 所有 epoch 中最好的模型对应的 acc 和 loss
        best_acc = {'train':0.0, 'valid':0.0}
        best_loss = {'train':0.0, 'valid':0.0}
        epoch_loss = {}
        epoch_acc = {}

        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch = 0
        # 设置优化器
        optimizer, scheduler = self.get_optimizer(model, self.args.optimizer, self.args.scheduler)  

        for epoch in range(self.args.epoch):
            self.pblog.info('Epoch: {:>3}/{:>3}   , lr={:0<.8f} , batch_size={} , model={}'.format(epoch, self.args.epoch-1, optimizer.param_groups[0]['lr'],
                        self.args.batch_size, self.args.model_type))
            # 每个 epoch 均设置了参数训练和验证
            for phase in ['train', 'valid']:
                
                if phase == 'train':
                    model.train()  # 将模型设置为参数可变
                    data_loader = dataloaders['train']
                else:
                    model.eval()   # 将模型设置为参数不可变，进行验证
                    data_loader = dataloaders['valid']

                dl_len = len(data_loader)   # dataloader 数据的大小
                running_corrects = 0.0    # 每个 epoch 的正确率
                running_sum = 0.0   # 表示已经处理完的图片数量
                loss_l = []
                loss_n = []
                
                with tqdm(total=dl_len, desc='Fold '+fold+" 's "+phase+'(%d/%d)'%(epoch,self.args.epoch), ncols=150) as _tqdm:   # 显示进度条
                    for inputs, labels, img_path in data_loader:
                        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

                        # 只有处于训练状态时才能调整参数
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            # _, preds = torch.max(outputs, 1)
                            loss = self.criterion(outputs, labels)
                            if phase == 'train':
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                               
                        # 评价本次预测结果
                        scores, preds = torch.max(outputs, 1)
                        running_corrects += torch.sum(preds == labels).item()
                        running_sum += labels.size(0)

                        # 保存训练过程中的数据    
                        loss_l.append(loss.item())
                        loss_n.append(outputs.size(0))
                        
                        _tqdm.update(1)

                loss_l = np.array(loss_l)
                loss_n = np.array(loss_n)
                epoch_loss[phase] = (loss_l * loss_n).sum()  / running_sum
                epoch_acc[phase] = running_corrects / running_sum
                # 将平均 loss 和 acc 结果输出到 tensorboard
                dict = {'Loss': epoch_loss[phase], 'Acc': epoch_acc[phase]}
                self.tblog.add_scalars('fold_'+fold+'_'+phase, dict, epoch)
                
                # 输出本 epoch 的结果信息
                self.pblog.info('{}-->  | Loss: {:0<5.5f} | Acc: {:0<5.3f}%'
                    .format(phase, epoch_loss[phase], epoch_acc[phase]*100))
                if phase == 'valid':
                    self.pblog.info('  ')
                    # 保存所有 epoch 中 准确率最高的模型
                    if epoch>35 and (epoch_acc['valid'] > best_acc['valid'] or 
                            (epoch_acc['valid'] == best_acc['valid'] and epoch_acc['train'] > best_acc['train'])):
                        best_acc = epoch_acc.copy()
                        best_loss = epoch_loss.copy()
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_epoch = epoch

            scheduler.step()

        time_elapsed = time.time() - since
        self.pblog.debug('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        self.pblog.debug('Best validat Acc: {:2f} %, epoch Loss: {:4f} in epoch{:>3}'
            .format(best_acc['valid']*100, best_loss['valid'], best_epoch))
        self.pblog.debug('with train Acc: {:2f} %, train Loss: {:4f}'.format(best_acc['train']*100, best_loss['train']))
        self.pblog.debug(' ')

        # 可视化网络结构
        self.tblog.add_graph(model, inputs)

        return best_model_wts, best_acc, best_loss, best_epoch

    def _evaluate_model(self, model, testloader, fold_num):

        all_preds = torch.tensor([])
        all_labels = torch.tensor([])
        all_scores = torch.tensor([])
        path_list = []
        error_list = []
        # get predictions and labels in testset
        model.eval()
        self.pblog.debug('Evaluating the best model......')
        with torch.no_grad():
            with tqdm(total=len(testloader), desc='Fold {}'.format(fold_num)+" 's evaluate ", ncols=150) as _tqdm:   # 显示进度条
                for inputs, labels, image_paths in testloader:
                    inputs = inputs.cuda(non_blocking=True)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    scores = softmax(outputs.detach().cpu(),1)
                    preds = preds.detach().cpu()
                    all_preds = torch.cat((all_preds.long(), preds.long()), 0)
                    all_labels = torch.cat((all_labels.long(), labels), 0)
                    all_scores = torch.cat((all_scores,scores), 0)
                    path_list.extend(image_paths)

                    _tqdm.update(1)

        # 将 confusion matrix 和 ROC curve 输出到 tensorboard
        cm_name = "Fold "+str(fold_num)+"'s "+self.args.model_type+" Confusion Matrix"
        cm_figure, tp, fp, fn, tn = tools.plot_confusion_matrix(all_labels, all_preds, self.args.class_names, cm_name)
        if self.tblog is not None:
            self.tblog.add_figure(cm_name, cm_figure)

        roc_name = "Fold "+str(fold_num)+"'s "+self.args.model_type+" ROC Curve"
        roc_auc, roc_figure, opt_threshold, opt_point = tools.plot_ROC_curve(all_labels, all_scores, self.args.class_num, roc_name)
        if self.tblog is not None:
            self.tblog.add_figure(roc_name, roc_figure)

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

        self.pblog.debug('Acc = {:0<5.5f} , auc={:0<5.5f}, optimize threshold={:0<5.5f} (fpr, tpr)={}'
            .format(acc, roc_auc, opt_threshold, opt_point))
        self.pblog.debug('Nonstandard Precision = {:0<5.5f} , Nonstandard Recall = {:0<5.5f} , Nonstandard Specificity={:0<5.5f} , Nonstandard F1 = {:0<5.5f}'
            .format(indicator_n['precision'], indicator_n['recall'], indicator_n['specificity'], indicator_n['f1_score']))
        self.pblog.debug('Standard Precision = {:0<5.5f} , Standard Recall = {:0<5.5f} , Standard Specificity={:0<5.5f} , Standard F1 = {:0<5.5f}'
            .format(indicator_s['precision'], indicator_s['recall'], indicator_s['specificity'], indicator_s['f1_score']))
        
        self.pblog.debug('=' * 50)
        self.pblog.info(' ')

        # 找出分类错误的图片的路径，以 图片路径+真实类标号 形式存储
        for index in torch.where(all_preds != all_labels)[0].tolist():
            error_list.append((path_list[index],all_labels[index]))
        
        return acc, roc_auc, opt_threshold, indicator_n, indicator_s, error_list, all_preds, all_scores, all_labels, path_list

    def kfold_train(self, args, dataset, model, random_seed):
        """
        本函数中会将训练集分层抽样成5部分,每次使用其中某一部分作为内部测试集,其余4部分再按照9:1划分训练集和验证集
        注意:本函数并未使用外部测试集
        """
        # 创建记录各折作为测试集时的预测结果
        all_labels = torch.tensor([])
        all_preds = torch.tensor([])
        all_scores = torch.tensor([])
        kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=random_seed)

        train_dataset = dataset['train']
        valid_dataset = dataset['valid'] # 此处的valid_dataset 与 train_dataset 的区别仅在于transform不同
        x, y = train_dataset.get_XY()

        init_dict = copy.deepcopy(model.state_dict())

        # 将评价指标写入表格
        record_dic = {}
        record_dic['name\\fold'] = ['test acc', 'test auc', 'opt_threshold', 'test n_precision', 'test n_recall', 'test n_specificity', 'test n_f1',
            'test s_precision', 'test s_recall', 'test s_specificity', 'test s_f1',
            'best valid acc', 'best valid loss', 'train acc', 'train loss', 'epoch index']
        # 将分类错误的图片信息写入表格
        csv_name = tools.gen_name(args.model_type+'_error','.csv')
        error_csv = open(join(args.sub_dir,csv_name), "w")
        error_writer = csv.writer(error_csv, dialect = "excel")

        for fold, (train_ids, test_ids) in enumerate(kfold.split(x,y)):
                
            # 再将训练集划分为训练集和选择参数的验证集
            ids_labels = np.array([y[i] for i in train_ids])
            train_ids, valid_ids, _, _  = train_test_split(train_ids, ids_labels, test_size=0.1, random_state=random_seed, stratify=ids_labels)

            self.pblog.info('Fold {}: valid indexs {}'.format(fold, valid_ids))
            self.pblog.info('Fold {}: test indexs {}'.format(fold, test_ids))
            self.pblog.info('-' * 50)

            train_subsampler = SubsetRandomSampler(train_ids)
            valid_subsampler = SubsetRandomSampler(valid_ids)
            test_subsampler = SubsetRandomSampler(test_ids)

            # trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
            trainloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_subsampler, num_workers=args.num_workers)
            validloader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_subsampler, num_workers=args.num_workers)
            # 注意此处dataloader加载的是验证集数据
            testloader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=test_subsampler, num_workers=args.num_workers)

            dataloaders = {'train':trainloader, 'valid':validloader, 'test':testloader}

            best_weights, best_acc, best_loss, epoch_idx = self._train(model, dataloaders, str(fold))
            
            # 保存数据和模型参数
            name = tools.gen_name(args.desc+'_fold_'+str(fold), '.pt')
            tools.save_model_dict(best_weights, args, name)

            # 评价模型
            model.load_state_dict(best_weights)
            acc, roc_auc, opt_threshold, indicator_n, indicator_s, error_list, pre_list, score_list, label_list, path_list = self._evaluate_model(model, dataloaders['test'], fold)
            record_dic[str(fold)] = [acc, roc_auc, opt_threshold, indicator_n['precision'], indicator_n['recall'], indicator_n['specificity'],indicator_n['f1_score'], 
                indicator_s['precision'], indicator_s['recall'], indicator_s['specificity'], indicator_s['f1_score'], 
                best_acc['valid'], best_loss['valid'], best_acc['train'], best_loss['train'], epoch_idx]
            
            path_list = [basename(dirname(path))+'-'+basename(path) for path in path_list]
            error_writer.writerow(['fold %d' % fold, path_list])

            error_writer.writerow(['Image Name', 'real label', 'path'])
            for img_path, real_label in error_list:
                if real_label.item() == 0:
                    error_writer.writerow([basename(img_path), 'nonstandard', img_path])
                else:
                    error_writer.writerow([basename(img_path), 'standard', img_path])
            error_writer.writerow([' '])

            # 重新载入初始权值，使得每个 fold 训练后的模型均不同
            model.load_state_dict(init_dict)

            # 保存本折测试集的预测结果及类标签
            all_preds = torch.cat((all_preds.long(), pre_list), 0)
            all_labels = torch.cat((all_labels.long(), label_list), 0)
            all_scores = torch.cat((all_scores, score_list), 0)

        # 汇总输出 k-fold 的结果
        cm_figure, tp, fp, fn, tn = tools.plot_confusion_matrix(all_labels, all_preds, self.args.class_names, 'Whole data Confusion Matrix')
        self.tblog.add_figure(args.model_type+"'s K-Fold Summary Confusion Matrix", cm_figure)
        all_acc = torch.sum(all_preds == all_labels).item() / len(all_labels)
        roc_name = "Summary 's "+self.args.model_type+" ROC Curve"
        all_auc, all_fig, all_opt_threshold, _ = tools.plot_ROC_curve(all_labels, all_scores, self.args.class_num, roc_name)
        # nonstandard 的评价指标
        all_indicator_n = {}
        all_indicator_n['precision'] = tp / (tp + fp)
        all_indicator_n['recall'] = tp / (tp + fn)
        all_indicator_n['specificity'] = tn / (tn + fp)
        if acc == 0. and indicator_n['recall'] == 0.:
            all_indicator_n['f1_score'] = float('nan')
        else:
            all_indicator_n['f1_score'] = 2 * (indicator_n['precision'] * indicator_n['recall']) / (indicator_n['precision'] + indicator_n['recall'])
        # standard 的评价指标
        all_indicator_s ={}
        all_indicator_s['precision'] = tn / (tn + fn)
        all_indicator_s['recall'] = tn / (tn + fp)
        all_indicator_s['specificity'] = tp / (tp + fn)
        if acc == 0. and indicator_s['recall'] == 0.:
            all_indicator_s['f1_score'] = float('nan')
        else:
            all_indicator_s['f1_score'] = 2 * (indicator_s['precision'] * indicator_s['recall']) / (indicator_s['precision'] + indicator_s['recall'])

        record_dic['summary'] = [all_acc, all_auc, all_opt_threshold, all_indicator_n['precision'], all_indicator_n['recall'], all_indicator_n['specificity'],all_indicator_n['f1_score'], 
                all_indicator_s['precision'], all_indicator_s['recall'], all_indicator_s['specificity'], all_indicator_s['f1_score'],'','','','','']
        
        self.pblog.debug('dataset evaluation: acc = {:0<5.5f}, auc = {:0<5.5f}'.format(all_acc, all_auc))
        self.pblog.debug('Nonstandard precison = {:0<5.5f} , Nonstandard recall = {:0<5.5f} , Nonstandard specificity = {:0<5.5f} , Nonstandard f1 = {:0<5.5f}'
            .format(all_indicator_n['precision'], all_indicator_n['recall'], all_indicator_n['specificity'], all_indicator_n['f1_score']))
        self.pblog.debug('Standard precison = {:0<5.5f} , Standard recall = {:0<5.5f} , Standard specificity = {:0<5.5f} , Standard f1 = {:0<5.5f}'
            .format(all_indicator_s['precision'], all_indicator_s['recall'], all_indicator_s['specificity'], all_indicator_s['f1_score']))

        csv_name = tools.gen_name(args.model_type+'_evaluate','.csv')
        pd.DataFrame(record_dic).to_csv(join(args.sub_dir,csv_name))
        error_csv.close()

    def single_train(self, args, dataset, model, random_seed, save_name=None):
        """
        本函数中会按9:1将训练集划分为训练集和验证,使用外部测试集做评估
        save_name:如果为None则不保存模型
        """
        # 创建记录各折作为测试集时的预测结果
        all_labels = torch.tensor([])
        all_preds = torch.tensor([])

        # 将评价指标写入表格
        record_dic = {}
        record_dic['name\\fold'] = ['test acc', 'test auc', 'opt_threshold', 'test n_precision', 'test n_recall', 'test n_specificity', 'test n_f1',
            'test s_precision', 'test s_recall', 'test s_specificity', 'test s_f1',
            'best valid acc', 'best valid loss', 'train acc', 'train loss', 'epoch index']
        
        train_dataset = dataset['train']
        valid_dataset = dataset['valid']
        test_dataset = dataset['test']
        x, y = train_dataset.get_XY()
        # 再将训练集划分为训练集和测试集
        train_ids, valid_ids, _, _  = train_test_split(list(range(len(y))), y, test_size=0.1, random_state=random_seed, stratify=y)

        self.pblog.info('Valid indexs {}'.format(valid_ids))
        self.pblog.info('-' * 50)

        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)

        # trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
        trainloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_subsampler, num_workers=args.num_workers)
        validloader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_subsampler, num_workers=args.num_workers)
        testloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        dataloaders = {'train':trainloader, 'valid':validloader, 'test':testloader}

        best_weights, best_acc, best_loss, epoch_idx = self._train(model, dataloaders, str(0))
        
        # 保存数据和模型参数
        if save_name is not None:
            name = tools.gen_name(save_name, '.pt')
            tools.save_model_dict(best_weights, args, name)

        acc, roc_auc, opt_threshold, indicator_n, indicator_s, error_list, pre_list, all_scores, label_list, path_list = self._evaluate_model(model, dataloaders['test'], 0)

        # 保存训练过程的指标
        record_dic[''] = [acc, roc_auc, opt_threshold, indicator_n['precision'], indicator_n['recall'], indicator_n['specificity'],indicator_n['f1_score'], 
                indicator_s['precision'], indicator_s['recall'], indicator_s['specificity'], indicator_s['f1_score'], 
                best_acc['valid'], best_loss['valid'], best_acc['train'], best_loss['train'], epoch_idx]

        csv_name = tools.gen_name(args.model_type+'_evaluate','.csv')
        pd.DataFrame(record_dic).to_csv(join(args.sub_dir,csv_name))

        csv_name = tools.gen_name(args.model_type+'_error','.csv')
        error_csv = open(join(args.sub_dir,csv_name), "w")
        error_writer = csv.writer(error_csv, dialect = "excel")
        path_list = [basename(dirname(path))+'-'+basename(path) for path in path_list]
        error_writer.writerow(['test path', path_list])
        error_writer.writerow(['Image Name', 'real label', 'path'])
        for img_path, real_label in error_list:
            if real_label.item() == 0:
                error_writer.writerow([basename(img_path), 'nonstandard', img_path])
            else:
                error_writer.writerow([basename(img_path), 'standard', img_path])
        error_writer.writerow([' '])

    def get_optimizer(self, model, optimizer_name, scheduler_name):
        optimizer, scheduler = None, None
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.sgd_m, weight_decay=self.args.sgd_w)
        else: 
            raise ValueError('No optimazer: {}'.format(self.args.optimizer))
        if scheduler_name == 'steplr':
            # 每 step_size 个 epoch 更新一次 lr
            scheduler = StepLR(optimizer, step_size=self.args.lr_epoch, gamma=0.1)
        elif scheduler_name == 'coslr':
            scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-9)
        elif scheduler_name == 'explr':
            scheduler = ExponentialLR(optimizer, gamma=0.98)
        else:
            raise ValueError('No scheduler: {}'.format(self.args.scheduler))
        return optimizer, scheduler