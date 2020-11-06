import torch
import sys
from torch import nn

# importo il path del modulo (da risistemare)
sys.path.append("/home/riccardo/git-sw/ann_modelli_importati/inceptionresnetv2_nuova/")
import pretrainedmodels as ptm

# cosi' posso caricare la rete in un colpo
nome_completo = "best.pth"

mm = torch.load(nome_completo, map_location='cpu')

print(mm)
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