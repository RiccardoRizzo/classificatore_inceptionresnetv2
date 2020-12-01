"""
Classificatore realizzato per caricare gli addestramenti di Filippo
e fornire in output le features esiderate.
Da usare anche con la FID
"""
import torch
import sys
from torch import nn

import matplotlib.pyplot as plt

import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Dataset
import torchvision.transforms as transforms

### SETTA IL DEVICE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# importo il path del modulo (da risistemare)
sys.path.append("/home/riccardo/git-sw/ann_modelli_importati/inceptionresnetv2_nuova/")
import pretrainedmodels as ptm

import dataset_pytorch as dspt

# cosi' posso caricare la rete in un colpo
nome_completo = "best.pth"

mm = torch.load(nome_completo, map_location=device)


print(mm)

## CODICE DA 
## https://www.kaggle.com/basu369victor/pytorch-tutorial-the-classification
## ========================================================================

## CREO IL DATASET
## ===============
BASE_PATH = "/home/riccardo/Focus/22==Esperimenti/Anno2020/classificatore_inception_resnet/sorgenti/dataset/"
subdirs = ["adenosis", "blabla"]
etichette = ["adenosis", "blabla"]

## DEFINISCE LA TRASFORMAZIONE
##============================
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
     )

ds = dspt.crea_dataset_immagini(BASE_PATH, 
                                subdirs, 
                                etichette, 
                                transform=transform)

## DEFINISCO I RAPPORTI FRA TRAINING E TEST 
## E IL BATCH SIZE
## =======================================
batch_size = 6
validation_split = .3
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
## EQUIVALENTE A 
# from sklearn.model_selection import train_test_split
# tr, val = train_test_split(data.label, stratify=data.label, test_size=0.1)
dataset_size = len(ds)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

## DEFINISCE I SAMPLERS
## ======================
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

## DEFINISCE I LOADERS
## ===================
train_loader = torch.utils.data.DataLoader(ds, 
                                            batch_size=batch_size, 
                                            sampler=train_sampler)

validation_loader = torch.utils.data.DataLoader(ds, 
                                            batch_size=batch_size,
                                            sampler=valid_sampler)






###################################################################
###################################################################
## INIZIO CLASSIFICAZIONE
###################################################################
###################################################################







## VISUALIZZAZIONE
##================
def img_display(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg



# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.size(), labels.size())
image_types = {0: 'adenosis', 1: 'blabla'}
# Viewing data examples used for training
fig, axis = plt.subplots(2, 2, figsize=(15, 10))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        image, label = images[i], labels[i]
        ax.imshow(img_display(image)) # add image
        ax.set(title = f"{image_types[label.item()]}") # add label

plt.show()


##############################################################
###
###  CODICE INUTILE
##############################################################


# Load the checkpoint file.
#state_dict = torch.load(nome_completo, map_location='cpu')

# Get the 'params' dictionary from the loaded state_dict.
#params = state_dict['model_state_dict']


#netG = Generator(params, nz, ngf, nc, k ).to(device)
# Load the trained generator weights.
#netG.load_state_dict(state_dict['model_state_dict'])
#print(netG)

""" 
num_classes = 8

model_name ='inceptionresnetv2'
#mm = ptm.__dict__[model_name](num_classes=1000,  pretrained='imagenet')
mm = ptm.__dict__[model_name](num_classes=1000, pretrained = None)

# ultimi strati fully connected
fv_last_linear = nn.Sequential(
            nn.Linear(1536, 740),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(740, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
            nn.LogSoftmax(dim=1))
        #fv_last_linear = nn.Sequential(nn.Linear(1536, 512),
         #               nn.ReLU(),
          #              nn.Dropout(0.3),
           #             nn.Linear(512, 128),
            #            nn.ReLU(),
             #           nn.Dropout(0.3),
              #          nn.Linear(128, 32),
               #         nn.ReLU(),
                #        nn.Dropout(0.3),
                 #       nn.Linear(32, num_classes),
                  #      nn.LogSoftmax(dim=1))
        #fv_last_linear.weight.data = model.last_linear.weight.data[1:]
        #fv_last_linear.bias.data = model.last_linear.bias.data[1:]

mm.last_linear = fv_last_linear
 """