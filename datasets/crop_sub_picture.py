# # 数据增广实验
# ## 1.读取图像
import torchvision
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
import torchvision.transforms as transforms
from PIL import Image
from datasets.myTransform import AddPepperNoise

# in_path = "./原始数据集/standard/正常1.bmp" # 较大尺寸的图片
in_path = "./括约肌重建（有时间轴）/合格/IMG_20210911_1 (8).bmp" # 较小尺寸图片
# in_path = "/media/huangyujun/disk/workspace/Altrasound_pro/seperated_dataset/standard/1/2.png"

img_w = 600
img_h = 350
output_shape = (img_w, img_h)


img = cv2.imread(in_path)
# print(img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

################################# 
# 处理未去除 1340x712 尺寸时间轴的图片
temp = gray[725:1255, 700]
if np.sum(temp) == 0:
    print('True!')
###################################

# 按照边缘截取图片
_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY) 
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
# points = [(400,230),(1000,230),(1560,230),(400,560),(1000,560),(1560,560),(400,940),(1000,940),(1560,940)] # 原始大图片
points = [(220,140), (510,140),(840,140),(220,300),(510,300),(840,300),(220,450),(510,450),(840,450)]
for idx in range(len(contours)):
    temp = img.copy()
    x,y,w,h = cv2.boundingRect(contours[idx])

    # crop = img[y:y+h,x:x+w]
    # print(idx)
    # print(crop.shape)
    # cv2.drawContours(temp,contours,idx,(0,0,255),3)
    # cv2.rectangle(temp, (x,y), (x+w,y+h), (0,255,0), 2)
    # plt.imshow(temp)
    # plt.show()
    
    # 选择合适的图片，并输出对应的序号
    # if w > 500 :
    if h > 130 and w >50 :
        crop = img[y:y+h,x:x+w]
        print('head point:({},{}) , picture shape:{} '.format(x,y,crop.shape))
        # print(idx)
        # for position, point in enumerate(points):
        #     if cv2.pointPolygonTest(contours[idx], point, False) != -1:
        #         print('head point:({},{}) , picture shape:{} , index:{}'.format(x,y,crop.shape,position))
        cv2.drawContours(temp,contours,idx,(0,0,255),3)
        cv2.rectangle(temp, (x,y), (x+w,y+h), (0,255,0), 2)
        plt.imshow(temp)
        plt.show()
    else:
        continue

# 检测是否路径已存在，若无则创建之
# def make_path(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# img_w = 608
# img_h = 355
# crop_imgs = []

# # 中间的三张图像
# crop_imgs.append(img[417-img_h:417, 30+img_w*2:30+img_w*3])
# crop_imgs.append(img[417:417+img_h, 30:30+img_w])
# crop_imgs.append(img[417:417+img_h, 30+img_w*1:30+img_w*2])
# crop_imgs.append(img[417:417+img_h, 30+img_w*2:30+img_w*3])

# temp = Image.fromarray(cv2.cvtColor(crop_imgs[0],cv2.COLOR_BGR2RGB))  
# tf = transforms.Compose([transforms.RandomCrop((img_h-20,img_w)), transforms.Resize(224)])


# for i in range(10):
# plt.subplot(1,2,1)
# print(img.shape)
# plt.imshow(img)
# plt.subplot(1,2,2)
# new_image = transforms.CenterCrop((250,400))(temp)
# new_image = AddPepperNoise(0.95, p=0.5)(temp)     
# new_image = transforms.RandomHorizontalFlip(p=0.2)(temp)
# new_image = transforms.RandomResizedCrop((img_h,img_w),scale=(0.9, 1.0))(temp)
# new_image = RandomCenterCrop(100, p=1.)(temp)
# new_img = tf(temp)
# print(new_image.size)
# plt.imshow(new_image)

# 根据边缘裁剪图片，写好的函数
def merge321(base_dir, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for item in os.listdir(base_dir):
        img_list = []   # 保存同一张图片中截取的图片
        bool_list = np.array([False]*9)
        # 获取原始图片的序号名，作为该图片截取图片的保存目录
        dir_name = re.search('\d+',item).group()
        save_path = os.path.join(save_dir,dir_name) # 截取图片的保存路径
        img = cv2.imread(os.path.join(base_dir,item))   # 这里无需将BGR转换为RGB，因为后面还要保存
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 将图片转化为灰度图，方便求得边缘
        _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)  # 转换为 0-1 图
        # 计算边缘
        binary, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
        for idx in range(len(contours)):
            # 获取包含轮廓的矩形
            x,y,w,h = cv2.boundingRect(contours[idx])
            if h > 300 and h <400 and w >550 and w < 700:   # 筛选轮廓条件
                crop = img[y:y+h,x:x+w]
                crop = cv2.resize(crop, output_shape)
                for position, point in enumerate(points):   # 确定图片序号
                    if cv2.pointPolygonTest(contours[idx], point, False) != -1:
                        img_list.append((position, crop))
                        bool_list[position] = True
            else:
                continue
      
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for position, crop in img_list:
            cv2.imwrite(os.path.join(save_path, str(position))+'.png', crop)
        judge = np.where(bool_list==False)[0]
        if len(judge) > 0:
            print(os.path.join(base_dir,item)+' without '+ str(judge))



