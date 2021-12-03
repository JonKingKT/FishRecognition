# This is a sample Python script.
import matplotlib_inline
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2 as cv
from PIL import Image
# from efficientnet.keras import EfficientNetB3
# import keras.backend as K
from matplotlib import pyplot as plt


from tqdm import tqdm

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from readImage import get_img

path = 'fish_data/'
img1=[]
img2=[]
img3=[]
img4=[]
img5=[]
img6=[]
img7=[]
img8=[]
img9=[]

def plot_y_train_hist():
  fig = plt.figure(figsize=(15,5))
  ax = fig.add_subplot(1,1,1)
  hist = ax.hist(Y_train,bins=n_classes)
  ax.set_title("the frequentcy of each category sign")
  ax.set_xlabel("bird")
  ax.set_ylabel("frequency")
  plt.show()
  return hist

hist = plot_y_train_hist()


def getImage():
    imgs=[]
    for i in tqdm(range(10)):
        # img1.append((str(i+1).zfill(5)+'.png',0))
        # img2.append((str(i+1).zfill(5)+'.png', 1))
        # img3.append((str(i+1).zfill(5)+'.png', 2))
        # img4.append((str(i+1).zfill(5)+'.png', 3))
        # img5.append((str(i+1).zfill(5)+'.png', 4))
        # img6.append((str(i+1).zfill(5)+'.png', 5))
        # img7.append((str(i+1).zfill(5)+'.png', 6))
        # img8.append((str(i+1).zfill(5)+'.png', 7))
        # img9.append((str(i+1).zfill(5)+'.png', 8))
        img_pil = get_img('fish_data/Black Sea Sprat_datasets/Black Sea Sprat/Black Sea Sprat/'+str(i+1).zfill(5)+'.png'
                          ,224,224)  # PIL.Image.Image对象
        imgs.append(img_pil)
        img_pil = get_img('fish_data/Gilt-Head Bream_datasets/Gilt-Head Bream/Gilt-Head Bream/' + str(i + 1).zfill(
                         5) + '.png',224,224)  # PIL.Image.Image对象
        imgs.append(img_pil)
        img_pil = get_img('fish_data/Hourse Mackerel_datasets/Hourse Mackerel/Hourse Mackerel/' + str(i + 1).zfill(
            5) + '.png',224,224)  # PIL.Image.Image对象
        imgs.append(img_pil)
        img_pil = get_img('fish_data/Red Mullet_datasets/Red Mullet/Red Mullet/' + str(i + 1).zfill(
            5) + '.png',224,224)  # PIL.Image.Image对象
        imgs.append(img_pil)
        img_pil = get_img('fish_data/Red Sea Bream_datasets/Red Sea Bream/Red Sea Bream/' + str(i + 1).zfill(
            5) + '.png',224,224)  # PIL.Image.Image对象
        imgs.append(img_pil)
        img_pil =get_img('fish_data/Sea Bass_datasets/Sea Bass/Sea Bass/' + str(i + 1).zfill(
            5) + '.png',224,224)  # PIL.Image.Image对象
        imgs.append(img_pil)
        img_pil = get_img('fish_data/Shrimp_datases/Shrimp/Shrimp/' + str(i + 1).zfill(
            5) + '.png',224,224)  # PIL.Image.Image对象
        imgs.append(img_pil)
        # img_pil = get_img('fish_data/Striped Red Mullet_datasets/Striped Red Mullet/Striped Red Mullet/' + str(i + 1).zfill(
        #     5) + '.png',224,224)
        # imgs.append(img_pil)
        img_pil =get_img('fish_data/Trout_datasets/Trout/Trout/' + str(i + 1).zfill(
            5) + '.png',224,224)  # PIL.Image.Image对象
        imgs.append(img_pil)
    # plt.imshow(imgs[0] / 255)  #显示图片
    # plt.show()
    imgs = np.array(imgs,np.float32)  # (H x W x C), [0, 255], RGB

    print("imgs:")
    print(imgs.shape)

    # img = Image.fromarray(img)
    transf = transforms.ToTensor()
    img_tensor = torch.from_numpy(imgs)
    return img_tensor





# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print("imgs_tensor:")
    print(getImage().shape)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
