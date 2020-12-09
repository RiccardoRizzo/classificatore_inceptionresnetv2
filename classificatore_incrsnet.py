"""
Classificatore realizzato per caricare gli addestramenti di Filippo
e fornire in output le features esiderate.
Da usare anche con la FID
"""
import torch
import sys
import os
from torch import nn
import yaml

import matplotlib.pyplot as plt

import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Dataset
import torchvision.transforms as transforms

import salva_txt_matrici as sm

import dataset_pytorch as dspt

### SETTA IL DEVICE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



## VISUALIZZAZIONE
##================
def img_display(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg

if __name__ == "__main__":

    file_conf = sys.argv[1]
    with open(file_conf, "r") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        PARAMS = yaml.load(file, Loader=yaml.FullLoader)

    NOME_COMPLETO_MODELLO = PARAMS["NOME_COMPLETO_MODELLO"]
    BASE_PATH_DATI_CLASSIFICAZIONE = PARAMS["BASE_PATH_DATI_CLASSIFICAZIONE"]
    PATH_ARCHITETTURA_MODELLO = PARAMS["PATH_ARCHITETTURA_MODELLO"]
    NOME_FILE_MATRICE_CONFUSIONE = PARAMS["NOME_MATRICE_CONFUSIONE"]

    print(NOME_COMPLETO_MODELLO)
    print(BASE_PATH_DATI_CLASSIFICAZIONE)
    print(PATH_ARCHITETTURA_MODELLO)

    # importo il path del modulo (da risistemare)
    sys.path.append(PATH_ARCHITETTURA_MODELLO)
    import pretrainedmodels as ptm

    # BASEPATH DATI DA CLASSIFICARE
    BASE_PATH = BASE_PATH_DATI_CLASSIFICAZIONE

    # cosi' posso caricare la rete in un colpo
    nome_completo = NOME_COMPLETO_MODELLO

    model = torch.load(nome_completo, map_location=device)

    print(model)

    ## CODICE DA 
    ## https://www.kaggle.com/basu369victor/pytorch-tutorial-the-classification
    ## ========================================================================

    ## CREO IL DATASET
    ## ===============
    # leggo le directory 
    __temp = os.listdir(BASE_PATH)
    subdirs = []
    for __o in __temp:
        if os.path.isdir(os.path.join(BASE_PATH,__o)):
            subdirs.append(__o)


    ## DEFINISCE LA TRASFORMAZIONE
    ##============================
    transform = transforms.Compose(
        [
        transforms.ToTensor() # trasforma in [0,1] i valori
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        )


    ds = dspt.crea_dataset_immagini(BASE_PATH, 
                                    subdirs,  
                                    transform=transform)

    ## DEFINISCO  IL BATCH SIZE
    ## =======================================
    batch_size = 1000

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
            # trasferisce sulla GPU se disponibile
            images, labels = images.to(device), labels.to(device)
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
            print(sm.matrice_a_stringa(matrice_confusione, "", ",\t" ) )
    # stampa finale
    print("matrice di confusione finale")
    print(matrice_confusione)
    nome_file = NOME_FILE_MATRICE_CONFUSIONE
    np.savetxt(nome_file, matrice_confusione, delimiter=', ', fmt='%d')
    #######################################################################
    #######################################################################



