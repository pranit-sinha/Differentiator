import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

def predict(path):
    image = Image.open(path)
    # print(np.array(image).shape)
    image = np.array(image)
    imageR = image[:,:,0]
    imageG = image[:,:,1]
    imageB = image[:,:,2]
    imageR = imageR.flatten()
    imageG = imageG.flatten()
    imageB = imageB.flatten()
    imageR_mean = np.mean(imageR)
    imageG_mean = np.mean(imageG)
    imageB_mean = np.mean(imageB)
    imageR_std = np.std(imageR)
    imageG_std = np.std(imageG)
    imageB_std = np.std(imageB)
    imageR_sum = np.sum(imageR)
    imageG_sum = np.sum(imageG)
    imageB_sum = np.sum(imageB)
    # print(f'Image Red Channel mean: {imageR_mean}, std: {imageR_std}, sum: {imageR_sum}')
    # print(f'Image Green Channel mean: {imageG_mean}, std: {imageG_std}, sum: {imageG_sum}')
    # print(f'Image Blue Channel mean: {imageB_mean}, std: {imageB_std}, sum: {imageB_sum}')

    total_sum = np.sum(image)
    tota_mean = np.mean(image)
    total_std = np.std(image)
    # print(f'Image total mean: {tota_mean}, std: {total_std}, sum: {total_sum}')
    return (imageR_mean, imageR_std, imageR_sum, imageG_mean, imageG_std, imageG_sum, imageB_mean, imageB_std, imageB_sum, tota_mean, total_std, total_sum,)


df = pd.DataFrame(columns=['R_mean', 'R_std', 'R_sum','G_mean', 'G_std', 'G_sum', 'B_mean', 'B_std', 'B_sum', 'total_mean', 'total_std', 'total_sum', 'class'])
paths = ["/Users/mraoaakash/Documents/research/research-tnbc/Differentiator/masterData/whiteSpace", "/Users/mraoaakash/Documents/research/research-tnbc/Differentiator/masterData/cellSpace"]
for path in paths:
    num = 0
    for img in os.listdir(path):
        prediction  = predict(os.path.join(path, img))
        prediction = (prediction[0], prediction[1], prediction[2], prediction[3], prediction[4], prediction[5], prediction[6], prediction[7], prediction[8], prediction[9], prediction[10], prediction[11], path.split("/")[-1])
        df1 = pd.DataFrame(prediction, index=['R_mean', 'R_std', 'R_sum','G_mean', 'G_std', 'G_sum', 'B_mean', 'B_std', 'B_sum', 'total_mean', 'total_std', 'total_sum', 'class'])
        df = df.append(df1.T, ignore_index=True)

df.to_csv('master.csv')