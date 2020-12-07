"""
Classificatore realizzato per caricare gli addestramenti di Filippo
e fornire in output le features esiderate.
Da usare anche con la FID
"""
import torch
import sys
import os
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

## VISUALIZZAZIONE
##================
def img_display(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg





# cosi' posso caricare la rete in un colpo
nome_completo = "/home/riccardo/Desktop/Link to classificatore_inception_resnet/best.pth"

model = torch.load(nome_completo, map_location=device)


print(model)

## CODICE DA 
## https://www.kaggle.com/basu369victor/pytorch-tutorial-the-classification
## ========================================================================

## CREO IL DATASET
## ===============
BASE_PATH = "/home/riccardo/Focus/22==Esperimenti/Anno2020/Articolo Workshop ICPR/GAN/dataset/train_tutte_dim_eguali/"

# leggo le directory 
__temp = os.listdir(BASE_PATH)
subdirs = []
for __o in __temp:
    if os.path.isdir(os.path.join(BASE_PATH,__o)):
        subdirs.append(__o)

#rint(subdirs)


## DEFINISCE LA TRASFORMAZIONE
##============================
transform = transforms.Compose(
    [
    transforms.ToTensor()
   # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    )


ds = dspt.crea_dataset_immagini(BASE_PATH, 
                                subdirs,  
                                transform=transform)

## DEFINISCO  IL BATCH SIZE
## =======================================
batch_size = 1000
shuffle_dataset = True
random_seed= 42

# IL TEST VA FATTO SU TUTTO IL DATASET
# val_indices contiene la lista degli indici
dataset_size = len(ds)
val_indices = list(range(dataset_size))

## DEFINISCE I SAMPLERS
## ======================
valid_sampler = SubsetRandomSampler(val_indices)

## DEFINISCE I LOADERS
## ===================
validation_loader = torch.utils.data.DataLoader(ds, 
                                            batch_size=batch_size,
                                            sampler=valid_sampler)

## INIZIO CLASSIFICAZIONE
#dataiter = iter(validation_loader)

#images, labels = dataiter.next()


matrice_confusione = np.zeros( (len(subdirs), len(subdirs)) ) 
num_elementi_totali = len(ds)
num_elementi_testati = 0

# comincia il test
with torch.no_grad():
    model.eval()
    # batch di valutazione
    for images, labels in validation_loader:

        num_elementi_testati += len(images)
        for image, label in zip(images, labels):
            image_tensor = image.unsqueeze_(0)
            output_ = model(image_tensor)
            output_ = output_.argmax()
            # la classe reale sta sulle righe
            r = label.item()
            # la predizione sulle colonne
            c = output_.item()  
            matrice_confusione[r][c] += 1

        # stampa report parziale
        print("num. elementi testati : " + str(num_elementi_testati) + " su " +str(num_elementi_totali))
        print(matrice_confusione)
# stampa finale
print("matrice di confusione finale")
print(matrice_confusione)
#######################################################################
#######################################################################



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