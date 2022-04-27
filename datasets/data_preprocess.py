from posix import listdir
import numpy as np
import os
import matplotlib.image as mpimg
import cv2
import re
from random import sample
import random
import numpy as np
from tqdm import tqdm

# 输出的图片尺寸
img_w = 600
img_h = 350
output_shape = (img_w, img_h)
# 一下列表的顺序是从上到下，从左到右，对9张图进行编号排列的
# point表示子图片的左上角坐标,crop_size表示裁剪的尺寸
# 原始图片尺寸为 1856x1132
points_1856x1132 = [(32,58), (640,58), (1248,58), (32,416), (640,416), 
        (1248,416), (32,774), (640,774), (1248,774)]
crop_size_1856x1132 = (606,356)
# 原始图片尺寸为 1000x528
point_1000x528 = [(32,58), (355,58), (678,58), (32,215), (355,215), 
        (678,215), (32,372), (355,372), (678,372)]
crop_size_1000x528 = (321,155)
# 原始图片尺寸为 1340x712
point_1340x712 = [(32,58),(468,58),(904,58),(32,276),(468,276),
        (904,276),(32,494),(468,494),(904,494)]
crop_size_1340x712 = (434,216)
# 原始图片尺寸为 1512x816(不合格719)
point_1512x816 = [(32,58),(526,58),(1020,58),(32,311),(526,311),(1020,311),(32,564),(526,564),(1020,564)]
crop_size_1512x816 = (492,251)
# 原始图片尺寸为 2300x1252（不合格567 568）
point_2300x1252 = []
crop_size_2300x1252 = (754,376)
# 原始图片尺寸为 1492x640(不合格574)
point_1492x640 = [(),(),()]
crop_size_1492x640 = (492,251)

# 设定随机种子
random.seed(100)

def cropImageSave(img_list, save_dir):
 
    for img_path in tqdm(img_list):
        # 因为有部分图片含有时间轴，需要单独处理
        time_bar = True
        # 获取原始图片的序号名，作为该图片截取图片的保存目录
        img_name = os.path.basename(img_path)
        dir_name = re.search('\d+',img_name).group()
        save_path = os.path.join(save_dir,dir_name) # 截取图片的保存路径
        img = cv2.imread(img_path)   # 这里无需将BGR转换为RGB，因为后面还要保存
        
        if img.shape[0] == 1132 and img.shape[1] == 1856:
            points = points_1856x1132
            crop_size = crop_size_1856x1132
        elif img.shape[0] == 528 and img.shape[1] == 1000:
            points = point_1000x528
            crop_size = crop_size_1000x528
        elif img.shape[0] == 712 and img.shape[1] == 1340:
            points = point_1340x712
            crop_size = crop_size_1340x712
            # 因为这批数据所有的 1340x712 尺寸图片均含时间轴
            # time_bar = True 
        elif img.shape[0] == 816 and img.shape[1] == 1512:
            points = point_1512x816
            crop_size = crop_size_1512x816
        else :
            print('error size in {} , img_size={}'.format(img_path,img.shape))
            continue
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for idx,(x,y) in enumerate(points):    
            w, h = crop_size
            # if time_bar and (idx==7 or idx==8):
            if time_bar :
                crop = img[y:y+h-18,x:x+w] # 18是时间轴的高度
            else:
                crop = img[y:y+h,x:x+w]
            crop = cv2.resize(crop, output_shape)
            cv2.imwrite(os.path.join(save_path, str(idx))+'.png', crop)
        
def cropImageSave_split(base_dir, save_base_dir, class_name, split_rate):

    train_save_dir = os.path.join(os.path.join(save_base_dir,'train'), class_name)
    test_save_dir = os.path.join(os.path.join(save_base_dir,'test'), class_name)

    if not os.path.exists(train_save_dir):
        os.makedirs(train_save_dir)
    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)

    file_list = list(os.path.join(base_dir,item) for item in listdir(base_dir))

    train_list = sample(file_list, int(len(file_list)*split_rate))
    valid_list = list(set(file_list).difference(train_list))
    cropImageSave(train_list,train_save_dir)
    cropImageSave(valid_list,test_save_dir)
    


if __name__ == '__main__':

    # base_dir = '原始数据集'
    # split_names = ["train", "valid"]
    split_rate = 0.9
    base_dir1 = '/media/huangyujun/disk/data/盆底质控数据/重建括约肌/合格/'
    base_dir2 = '/media/huangyujun/disk/data/盆底质控数据/重建括约肌/不合格/'

    # cropImageSave(base_dir1, './seperated_dataset','standard')
    # cropImageSave(base_dir2, './seperated_dataset','nonstandard')
    print('*'*10+'processing standard images'+'*'*10)
    cropImageSave_split(base_dir1, '/media/huangyujun/disk/data/盆底质控数据/splited_dataset_noT', 'standard', split_rate)
    print('*'*10+'processing nonstandard images'+'*'*10)
    cropImageSave_split(base_dir2, '/media/huangyujun/disk/data/盆底质控数据/splited_dataset_noT', 'nonstandard', split_rate)



