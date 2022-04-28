# @ FileName:main.py
# @ Author: Yujun
# @ Time: 21-8-2 下午2:34
from os import makedirs
from os.path import exists, join, splitext
from posix import listdir
from posixpath import basename
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import torch
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from datetime import datetime
from sklearn.preprocessing import label_binarize
from PIL import Image
import imageio
from torchvision import transforms
from datasets.dataset import mean_RGB, std_RGB, mean_gray, std_gray

time_mark = None

def check_mkdir(path):
    """
    输入:path
    输出:路径是否存在
    作用:检查路径是否存在,如不存在则创建该路径
    """
    result = exists(path)
    if not result:
        makedirs(path)
    return result

def plot_confusion_matrix(all_labels, all_preds, class_names, title):
    """
    输入:cm (array, shape = [n, n]):混淆矩阵
        class_names (array, shape = [n]):分类任务中类别的名字
        title (string):生成图片的标题
    输出:figure:混淆矩阵可视化图片对象
    作用:生成混淆矩阵可视化图片,返回不合格的 TP、FP、FN、TN
    """
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())
    figure = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.array(range(len(class_names)))    
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks, class_names, rotation=-45)

    # 将混淆矩阵的数值标准化(保留小数点后两位),即每一个元素除以矩阵中每一行(真实标签)元素之和
    # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    thresh = cm.max() / 2.
    # 此处遍历为按照生成的笛卡尔集顺序遍历
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment='center',
                color='white' if cm[i,j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure, cm[0][0], cm[1][0], cm[0][1], cm[1][1]

def plot_ROC_curve(all_labels, all_scores, num_classes, title):
    """
    输入:all_labels:数据的真实标签
        all_scores:输入数据的预测结果
        title:画出 ROC 图像的标题
    输出:figure:ROC曲线图像
    作用:绘制 ROC 曲线并计算 AUC 值
    """
    # 需注意绘制 ROC 曲线时,传入的 all_labels 必须转换为独热编码,all_socres 要转换为1维,元素代表取得评估概率（大于阈值为正例,否则为负例）
    figure = plt.figure()
    all_scores = torch.softmax(all_scores, dim=1)
    if all_labels.ndim != 1: # 多分类的情况
        binary_label = label_binarize(all_labels, classes=list(range(num_classes)))
        # 待修改,可参考 https://blog.csdn.net/u013347145/article/details/104332094
        pass
    else:
        all_labels, all_scores = all_labels.numpy(), all_scores.numpy()
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores[:,1])   
        roc_auc = auc(fpr, tpr)
        opt_idx = np.argmax(fpr-tpr)
        opt_threshold = thresholds[opt_idx]
        opt_point = (fpr[opt_idx], tpr[opt_idx])

        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标,真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
    
    return roc_auc, figure, opt_threshold, opt_point

def save_model_dict(model_dict, args, save_name):
    """
    输入:model:需要保存的模型
        args:参数列表
        output_path(str):模型保存的位置
    输出:无
    作用:保存模型
    """
    save_dict = {'model_type':args.model_type,
                'model_dic':model_dict,
                'img_size':args.img_size,
                }

    torch.save(save_dict, join(args.model_dir, save_name))

def gen_name(desc, ext):
    # 每次执行,使用相同的 time_mark
    global time_mark
    if time_mark == None:
        time_mark = datetime.now().strftime('_%y%m%d_%H%M%S')
    return desc+time_mark+ext

def gen_grad_cam(model, img_dir, select_list, output_base_dir, index, is_gray=False):
    """
    输入:model:需要生成 CAM 的模型
        img_dir:一张原图切割出的9张图片保存路径
        
    """
    if is_gray is False:
        tf = transforms.Compose([
                        # transforms.Resize([224, 224]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean_RGB, std_RGB)])
    else:
        tf = transforms.Compose([
                        # transforms.Resize([224, 224]),
                        transforms.Grayscale(num_output_channels=1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean_gray, std_gray)])
    target_layer = model.get_feature_layer()
    cam = GradCAM(model=model,
                target_layers=[target_layer],
                use_cuda=True)
    tensor_list = []
    image_list = []
    result_list = []
    name_list = listdir(img_dir)
    name_list.sort()
    output_dir = join(output_base_dir,'model_%d' % index)
    check_mkdir(output_dir)
    for img_name in name_list:
        img_position, extension = splitext(img_name)
        if int(img_position) in select_list:
            img = Image.open(join(img_dir, img_name))
            image_list.append(img)
            tensor_list.append(tf(img))
    
    input_tensors = torch.cat(tensor_list, dim=0)
    input_tensors = torch.unsqueeze(input_tensors, dim=0)
    # grayscale_cam = cam(input_tensor=input_tensors, target_category=None)
    grayscale_cam = cam(input_tensor=input_tensors)
    
    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0,:]

    for idx,img in enumerate(image_list):
        img = img.resize((600,350))
        # img = img.resize((224,224))
        img = np.float32(img) / 255
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        # 首先保存图片
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(join(output_dir,str(idx)+'.png'), cam_image)
        
        # 接着生成视频
        cv2.putText(cam_image, 'model %d picuture %d'% (index,idx), (80, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                  color=(123,222,238), thickness=1, lineType=cv2.LINE_AA)
        # 使用原图大小,即不进行resize
        # cv2.putText(cam_image, 'fold %d picuture %d' % (index,idx), (400, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, 
        #           color=(123,222,238), thickness=2, lineType=cv2.LINE_AA)
        
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
        result_list.append(cam_image)
        # plt.imshow(cam_image)
        # plt.show()
        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        
    imageio.mimsave(join(output_dir, 'output.gif'),result_list,fps=1)

def load_model_dict(model_dir):
    """
    输入:args:参数列表
    输出: 无
    作用:自动载入模型
    """
    model_list = []
    model_names = listdir(model_dir)
    model_names.sort()
    if len(model_names) == 0:
        raise ValueError('No models to load!')
    else:
        for name in model_names:
            model_list.append(torch.load(join(model_dir,name)))
        return model_list

