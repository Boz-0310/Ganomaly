# In[]
import os
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image

gray_flag= 1  #是否灰階
img_width =  481 #圖片統一size
img_height =  271 #圖片統一size
train_path = 'train'  #訓練目錄
category_name=os.listdir(train_path)
nub_train = len(glob(train_path + '/*/*.bmp'))
#先生成空array，然後往裡面填每張圖片的array
if gray_flag == 1:
    all_data = np.zeros((nub_train,img_height,img_width),dtype=np.uint8) 
else:
    all_data = np.zeros((nub_train,img_height,img_width,3),dtype=np.uint8) 
all_y = np.zeros((nub_train,),dtype=np.uint8)

num=len(os.listdir(train_path+"/"+category_name[0]))

i = 0
for img_path in tqdm(glob(train_path + '/*/*.bmp')):
    if gray_flag == 1:
        img = Image.open(img_path).convert('L')
    else:   
        img = Image.open(img_path)
    img = img.resize((img_width,img_height)) #圖片resize
    arr = np.asarray(img)  #圖片轉array
    if gray_flag == 1:
        all_data[i, :, :] = arr #賦值
    else:   
        all_data[i, :, :, :] = arr #賦值
    
    if i>=num:
        all_y[i] = 0  #第一個類別設0
    else:
        all_y[i] = 1  #第二個類別設1
    i += 1

normal_x= all_data[all_y==0]  #全部的正常
normal_y= all_y[all_y==0]  #全部的正常
np.random.shuffle(normal_x) #資料打亂

mura_x= all_data[all_y !=0]  #全部的不正常
mura_y= all_y[all_y !=0]  #全部的不正常
np.random.shuffle(mura_x) #資料打亂

x_Test= np.zeros((100,img_height,img_width),dtype=np.uint8) 

x_Test[0:49,:,:]=mura_x[0:49,:,:]
x_Test[50:99,:,:]=normal_x[550:599,:,:]
y_Test=np.concatenate([mura_y[0:50],normal_y[550:600]]) #

#測試資料打散
per = np.random.permutation(x_Test.shape[0])
new_x_Test = x_Test[per, :, :]
new_y_Test = y_Test[per]

custom_train=(normal_x[0:549,:,:],normal_y[0:549])
custom_test=(new_x_Test,new_y_Test)
custom_data=(custom_train,custom_test)

# %%
