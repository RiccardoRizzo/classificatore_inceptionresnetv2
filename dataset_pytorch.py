import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
import torch
# Neural networks can be constructed using the torch.nn package.
""" 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
"""
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms



# SORGENTE PER COSTRUIRE IL DATALOADER
# https://www.kaggle.com/basu369victor/pytorch-tutorial-the-classification


# CREA LA CALSSE RELATIVA ALLA STRUTTURA DATI
class Image_Dataset(Dataset):
    def __init__(self, img_data, img_path,  img_size, transform=None):
        """
        img_data : dataset pandas contenente i nomi delle immagini
        img_path : base path per le immagini
        img_subdirs : lista delle sottodirectory delle immagini
        img_size : nuova dimensione delle immagini
        transform : eventuale altra trasformazione da fare sulle immagini
        """
        self.img_path = img_path
        self.transform = transform
        self.img_data = img_data
        self.resize = img_size
        
    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, index):


        img_name = os.path.join(self.img_path,self.img_data.loc[index, 'labels'],
                                self.img_data.loc[index, 'Images'])
        image = Image.open(img_name)
        #image = image.convert('RGB')
        image = image.resize(self.resize)
        label = torch.tensor(self.img_data.loc[index, 'encoded_labels'])
        if self.transform is not None:
            image = self.transform(image)
        return image, label




def crea_dataset_immagini(BASE_PATH, subdirs, etichette, transform = None):
    image=[]
    labels=[]

    # CREA UNA LISTA DI IMMAGINI ED UNA LISTA DI ETICHETTE
    # scandisce la directory delle immagini e per ogni sottodirectory
    # crea una classe
    for file in os.listdir(BASE_PATH):
        for dir in subdirs:
            # trova l'indice di dir per la etichetta
            ind_eti = subdirs.index(dir)
            if file == dir :
                for c in os.listdir(os.path.join(BASE_PATH, file)):
                    if c!='annotations':
                        image.append(c)
                        labels.append(etichette[ind_eti])

    # A PARTIRE DALLE LISTE CREA UN DATASET 
    data = {'Images':image, 'labels':labels} 
    data = pd.DataFrame(data) 
    #print(data.head)

    # CODIFICA LE ETICHETTE COME INTERI
    lb = LabelEncoder()
    data['encoded_labels'] = lb.fit_transform(data['labels'])
    #print(data.head)

    dataset = Image_Dataset(data, BASE_PATH,  (100, 100), transform )

    return dataset



# CONTROLLO IL CONTENUTO NEL DATALOADER

if __name__ == "__main__":
    # CARICO IL DATASET DI IMMAGINI DA CLASSIFICARE
    BASE_PATH = "/home/riccardo/Desktop/Link to classificatore_inception_resnet/sorgenti/dataset/"
    subdirs = ["adenosis", "blabla"]
    etichette = ["adenosis", "blabla"]

    crea_dataset_immagini(BASE_PATH, subdirs, etichette)
