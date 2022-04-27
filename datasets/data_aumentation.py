# # 数据增广实验
# ## 1.读取图像
import torchvision
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

in_path = "./原始数据集/图像-标准/正常1.bmp"

out_path = './数据增强/'

img = cv2.imread(in_path)

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

img_w = 608
img_h = 355
crop_imgs = []

# 中间的三张图像
crop_imgs.append(img[417:417+img_h, 30:30+img_w])
crop_imgs.append(img[417:417+img_h, 30+img_w*1:30+img_w*2])
crop_imgs.append(img[417:417+img_h, 30+img_w*2:30+img_w*3])

path = os.path.join(out_path,'原图')
make_path(path)

for i in range(0,3):
    cv2.imwrite(os.path.join(path,str(i+1)+'.png'),crop_imgs[i])

# ## 2.加入随机椒盐噪声
# path = os.path.join(out_path,'随机椒盐噪声')
# make_path(path)

# for i in range(0,3):
#     image_c = crop_imgs[i].copy()
#     prob = 0.05
#     black = np.array([0, 0, 0], dtype='uint8')
#     white = np.array([255, 255, 255], dtype='uint8')
#     probs = np.random.random(image_c.shape[:2]) # 形成与图片大小一样的随机矩阵
#     image_c[probs < (prob / 2)] = black
#     image_c[probs > 1 - (prob / 2)] = white

#     cv2.imwrite(os.path.join(path,str(i)+'.png'),image_c)

#     plt.subplot(3,2,i*2+1)
#     plt.imshow(crop_imgs[i-1])
#     plt.subplot(3,2,i*2+2)
#     plt.imshow(image_c)

# plt.savefig(path+'.png')
# plt.clf()

# ## 3. 随机翻转
# 水平翻转
# path = os.path.join(out_path,'随机水平翻转')
# make_path(path)

# for i in range(0,3):
#     image_c = crop_imgs[i].copy()
#     h_flip = cv2.flip(image_c,1)

#     cv2.imwrite(os.path.join(path,str(i)+'.png'),h_flip)

#     plt.subplot(3,2,i*2+1)
#     plt.imshow(crop_imgs[i-1])
#     plt.subplot(3,2,i*2+2)
#     plt.imshow(h_flip)

# plt.savefig(path+'.png')
# plt.clf()

# 垂直翻转
# path = os.path.join(out_path,'随机垂直翻转')
# make_path(path)

# for i in range(0,3):
#     image_c = crop_imgs[i].copy()
#     v_flip = cv2.flip(image_c,0)

#     cv2.imwrite(os.path.join(path,str(i+1)+'.png'),v_flip)

#     plt.subplot(3,2,i*2+1)
#     plt.imshow(crop_imgs[i-1])
#     plt.subplot(3,2,i*2+2)
#     plt.imshow(v_flip)

# plt.savefig(path+'.png')
# plt.clf()

# ## 4. 随机旋转
# path = os.path.join(out_path,'随机旋转')
# make_path(path)

# angle = np.random.randint(-10,10)
# for i in range(0,3):
#     image_c = crop_imgs[i-1].copy()
#     col, row = image_c.shape[:2]
#     M=cv2.getRotationMatrix2D((col/2,row/2),angle,1)
#     dst=cv2.warpAffine(image_c,M,(row,col))

#     cv2.imwrite(os.path.join(path,str(i+1)+'.png'),dst)

#     plt.subplot(3,2,i*2+1)
#     plt.imshow(crop_imgs[i-1])
#     plt.subplot(3,2,i*2+2)
#     plt.imshow(dst)

# plt.savefig(path+'.png')
# plt.show()
# plt.clf()

# ## 5.随机仿射变换
# path = os.path.join(out_path,'随机仿射变换')
# make_path(path)

# point1=np.float32([[50,50],[300,50],[50,200]])
# point2=np.float32([[10,100],[300,50],[100,250]])

# for i in range(0,3):
#     image_c = crop_imgs[i].copy()
#     col, row = image_c.shape[:2]
#     M=cv2.getAffineTransform(point1,point2)
#     dst=cv2.warpAffine(image_c,M,(row,col),borderValue=(0,0,0))

#     cv2.imwrite(os.path.join(path,str(i+1)+'.png'),dst)

#     plt.subplot(3,2,i*2+1)
#     plt.imshow(crop_imgs[i-1])
#     plt.subplot(3,2,i*2+2)
#     plt.imshow(dst)

# plt.savefig(path+'.png')
# plt.clf()

# ## 6. 随机缩放
# path = os.path.join(out_path,'随机缩放')
# make_path(path)

# factor = float(np.random.random_sample()*2)
# for i in range(0,3):
#     image_c = crop_imgs[i].copy()
#     height, width = image_c.shape[:2]
#     dst = cv2.resize(image_c,(int(factor*width),int(factor*height)))
#     cv2.imwrite(os.path.join(path,str(i+1)+'.png'),dst)

#     plt.subplot(3,2,i*2+1)
#     plt.imshow(crop_imgs[i-1])
#     plt.subplot(3,2,i*2+2)
#     plt.imshow(dst)

# plt.savefig(path+'.png')
# plt.show()
# plt.clf()

# # ## 7.随机亮度
# path = os.path.join(out_path,'随机亮度')
# make_path(path)
# # value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
# value = 50
# for i in range(0,3):
#     image_c = crop_imgs[i].copy()
#     hsv = cv2.cvtColor(image_c, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     if value >= 0:
#         lim = 255 - value
#         v[v > lim] = 255
#         v[v <= lim] += value
#     else:
#         lim = np.absolute(value)
#         v[v < lim] = 0
#         v[v >= lim] -= np.absolute(value)

#     final_hsv = cv2.merge((h, s, v))
#     dst = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#     cv2.imwrite(os.path.join(path,str(i+1)+'.png'),dst)

#     plt.subplot(3,2,i*2+1)
#     plt.imshow(crop_imgs[i-1])
#     plt.subplot(3,2,i*2+2)
#     plt.imshow(dst)

# plt.savefig(path+'.png')
# plt.clf()

# # ## 7.随机饱和度
# path = os.path.join(out_path,'随机饱和度')
# make_path(path)
# # value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
# value = 50
# for i in range(0,3):
#     image_c = crop_imgs[i].copy()
#     hsv = cv2.cvtColor(image_c, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     if value >= 0:
#         lim = 255 - value
#         s[s > lim] = 255
#         s[s <= lim] += value
#     else:
#         lim = np.absolute(value)
#         s[s < lim] = 0
#         s[s >= lim] -= np.absolute(value)

#     final_hsv = cv2.merge((h, s, v))
#     dst = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#     cv2.imwrite(os.path.join(path,str(i+1)+'.png'),dst)

#     plt.subplot(3,2,i*2+1)
#     plt.imshow(crop_imgs[i-1])
#     plt.subplot(3,2,i*2+2)
#     plt.imshow(dst)

# plt.savefig(path+'.png')
# plt.clf()

# # ## 8.随机对比度
# path = os.path.join(out_path,'随机对比度')
# make_path(path)
# value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
# for i in range(0,3):
#     image_c = crop_imgs[i-1].copy()
#     brightness = 10
#     contrast = random.randint(40, 100)
#     dummy = np.int16(image_c)
#     dummy = dummy * (contrast/127+1) - contrast + brightness
#     dummy = np.clip(dummy, 0, 255)
#     dst = np.uint8(dummy)
#     cv2.imwrite(os.path.join(path,str(i+1)+'.png'),dst)

#     plt.subplot(3,2,i*2+1)
#     plt.imshow(crop_imgs[i-1])
#     plt.subplot(3,2,i*2+2)
#     plt.imshow(dst)

# plt.savefig(path+'.png')
# plt.clf()
