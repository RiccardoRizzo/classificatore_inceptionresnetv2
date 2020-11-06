from __future__ import print_function, division, absolute_import
import argparse

from PIL import Image
import torch
import torchvision.transforms as transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import sys
sys.path.append('.')
import pretrainedmodels
import pretrainedmodels.utils as utils

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import time
import os.path

main_dir = '/home/filippo/Workshop/Segmentation/Inception_Res/pretrained-models.pytorch/examples/'

data_dir = '/home/filippo/Data/Frucci_Brancati/BreakHis/Four_Categories_A/'#100 samples for cat
data_dir = '/home/filippo/Data/Frucci_Brancati/BreakHis/Four_categories/'
data_dir = '/home/filippo/Data/Frucci_Brancati/BreakHis/Eight_categories//Eight_categories/'
#deep directory----------------
main_dir = '/home/fvella/Workshop/Inception_ResNet/pretrained-models.pytorch/examples/'
data_dir = '/home/ricrizzo/Focus/Datasets/Immagini/BreakHis_460'
data_dir = '/home/fvella/Data/Break_his/Eight_categories_Riccardo/'
#end deep directory----------------

#22 Settembre 2020
#changed data dir
main_dir = '/home/filippo/Workshop/Segmentation/Inception_Res/pretrained-models.pytorch/examples/'
main_dir = './'
data_dir = '/home/fvella/Data/Break_his/Four_categories/Linked_files/'
data_dir = '/home/fvella/Data/Break_his/Four_categories/Linked_files/'
data_dir = '/home/fvella/Data/Break_his/Linked_files_8Cats/'

#5 october Trapani ricreato set 100x : verifica dei dati di riccardo
data_dir = '/home/fvella/Data/Break_his/Eight_Categories_100/'

#15 october palermo dataset augmented
data_dir = "/home2/ricrizzo/Focus/22\=\=Esperimenti/Anno2020/Articolo\ Workshop\ ICPR/dataset_real_fake/"
data_dir = "/home/fvella/Data/Break_his/Data_Real_Fake/"

#17 oct 2020 palermo split dataset ...change valid dir
data_dir = "/home/fvella/Data/Break_his/Split_Eight_Cats_200/"

#22 oct 2020 create a baseline
data_dir = "/home/fvella/Data/Break_his/Linked_tt_valid_bare/"

model_names = sorted(name for name in pretrainedmodels.__dict__
    if not name.startswith("__")
    and name.islower()
    and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='inceptionresnetv2',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: inceptionresnetv2)',
                    nargs='+')
parser.add_argument('--path_img', type=str, default='data/cat.jpg')


def load_all_train_test_val(datadir, valid_size = .15, test_size = .2):
 #   train_transforms = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),])
 #   test_transforms = transforms.Compose([transforms.Resize(224),transforms.ToTensor(), ])
    dbg_load=True
    Resize_Flag =1
    Crop_Flag =0
    if(Crop_Flag):
        train_transforms = transforms.Compose([transforms.CenterCrop((400,300)),
                                          transforms.ToTensor(),
                                          ])

        val_transforms = transforms.Compose([transforms.CenterCrop((400,300)),
                                       transforms.ToTensor(),
                                       ])
                                       
        test_transforms = transforms.Compose([transforms.CenterCrop((400,300)),
                                      transforms.ToTensor(),
                                      ])
    if(Resize_Flag):
        #train_transforms = transforms.Compose([transforms.Resize(345),transforms.CenterCrop((525, 345)),transforms.ToTensor(),])
        #train_transforms = transforms.Compose([transforms.Resize(345),transforms.CenterCrop((525, 345)), transforms.ToTensor(),])
        train_transforms = transforms.Compose([transforms.Resize(345),
                                               transforms.CenterCrop((525, 345)),
                                               transforms.ToTensor(),])

        val_transforms = transforms.Compose([transforms.Resize(345),
                                               transforms.CenterCrop((525, 345)),
                                       transforms.ToTensor(),
                                       ])
        test_transforms = transforms.Compose([transforms.Resize(345),
                                               transforms.CenterCrop((525, 345)),
                                      transforms.ToTensor(),
                                      ])
                                      
    train_datadir = datadir + "/Train"
    val_datadir =  datadir + "/Valid"
    test_datadir =  datadir + "/Test"
    
    if not os.path.exists(train_datadir):
        train_datadir = datadir + "/train"
    if not os.path.exists(val_datadir):
        val_datadir =  datadir + "/valid"
    if not os.path.exists(test_datadir):
        test_datadir =  datadir + "/test"


    if(dbg_load):
        print("train_datadir = ", train_datadir)
        print("val_datadir = ", val_datadir)
        print ("test_datadir= ", test_datadir)
    
    train_data = datasets.ImageFolder(train_datadir, transform=train_transforms)
    val_data = datasets.ImageFolder(train_datadir,transform=test_transforms)
    test_data = datasets.ImageFolder(test_datadir, transform=test_transforms)

    num_train = len(train_data)
    num_test = len(test_data)
    indices = list(range(num_train))
    
    if(dbg_load):
        print("load_all)num train = ", num_train)
        print("load_all)num test = ", num_test)
        print("load_all)num val  = ", len(val_data))
        print ("load_all)test data= ", test_data)

    split_1 = int(np.floor(valid_size * num_train))
    #split_2 = int(np.floor(test_size * num_train)) + split_1

    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    
    val_idx = indices[:split_1]
    train_idx = indices[split_1:]
    test_idx = np.arange(0, num_test)


    #train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler =  SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    if(dbg_load):
        print("dbg_load]load_all)num test = ", num_test)
        print("load_all)len test sampler  = ", len(test_sampler))
        print("load_all)test sampler  = ", test_sampler)
    
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=16)
    testloader  = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=16)
    validloader = torch.utils.data.DataLoader(val_data, sampler=val_sampler, batch_size=16)
    
    if(dbg_load):
        print("dbg_load)split 1= ", split_1)
        print("num train sampler= ", len(train_sampler))
        print("num test sampler= ", len(test_sampler))
        print("load_all2)num val  sampler= ", len(val_sampler))
        print("val sampler = ", val_sampler)
        
        print("testloader = ", testloader)
        print("testloader dataset = ", testloader.dataset)

    return trainloader, validloader, testloader



def load_split_train_test_val(datadir, valid_size = .15, test_size = .2):
 #   train_transforms = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),])
 #   test_transforms = transforms.Compose([transforms.Resize(224),transforms.ToTensor(), ])
    dbg_load=True
    Resize_Flag =1
    Crop_Flag =0
    if(Crop_Flag):
        train_transforms = transforms.Compose([transforms.CenterCrop((400,300)),
                                          transforms.ToTensor(),
                                          ])

        val_transforms = transforms.Compose([transforms.CenterCrop((400,300)),
                                       transforms.ToTensor(),
                                       ])
        test_transforms = transforms.Compose([transforms.CenterCrop((400,300)),
                                      transforms.ToTensor(),
                                      ])
    if(Resize_Flag):
        train_transforms = transforms.Compose([transforms.Resize(345),
                                               transforms.CenterCrop((525, 345)),
                                          transforms.ToTensor(),
                                          ])

        val_transforms = transforms.Compose([transforms.Resize(345),
                                               transforms.CenterCrop((525, 345)),
                                       transforms.ToTensor(),
                                       ])
        test_transforms = transforms.Compose([transforms.Resize(345),
                                               transforms.CenterCrop((525, 345)),
                                      transforms.ToTensor(),
                                      ])

    train_data = datasets.ImageFolder(datadir,
                    transform=train_transforms)

    val_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)

    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)

    num_train = len(train_data)
    num_test = len(test_data)
    num_val = len(val_data)
    if(dbg_load):
        print("num_train:", num_train)
        print("num_test:", num_train)
        print("num_val:", num_train)
    indices = list(range(num_train))

    split_1 = int(np.floor(valid_size * num_train))
    split_2 = int(np.floor(test_size * num_train)) + split_1
    if(dbg_load):
        print("split_1:", split_1)
        print("split_2:", split_2)

    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    val_idx = indices[:split_1]
    test_idx =indices[split_1:split_2]
    train_idx = indices[split_2:]
    #train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler =  SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,sampler=train_sampler, batch_size=16)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=16)
    validloader = torch.utils.data.DataLoader(val_data, sampler=val_sampler, batch_size=16)
    return trainloader, validloader, testloader


def main():
    dbg_run = True
    dbg_run_deep = False
    dbg_valid_deep = True
    save_Valid_conf = False
    
    global args
    args = parser.parse_args()

    #trainloader, validloader, testloader =  load_split_train_test_val(data_dir)
    trainloader, validloader, testloader =  load_all_train_test_val(data_dir)

    print("len train =", len(trainloader), "shape:", np.shape(trainloader))
    print("len test =",  len(testloader), "shape:",np.shape(testloader))
    print("len test =",  len(validloader), "shape:",np.shape(validloader))
    print(trainloader.dataset.classes)
    num_classes =  len(trainloader.dataset.classes)
    print("num classes = ", num_classes)

    print("data_dir = ", data_dir)
    print("main_dir = ", main_dir)

    save_dir = main_dir + 'save/'
    if not os.path.exists(save_dir):
              os.makedirs(save_dir)


    device = torch.device("cuda" if torch.cuda.is_available()
                                      else "cpu")

    #device = "cpu"
    for arch in args.arch:
        # Load Model

        model_name ='inceptionresnetv2'

        model = pretrainedmodels.__dict__[model_name](num_classes=1000,
                                                pretrained='imagenet')

        
        #ALSO features are learned
        for q in model.parameters():
            q.requires_grad = True
        #    q.requires_grad = False
            
        for q in model.last_linear.parameters():
            q.requires_grad = True
            
        
        fv_last_linear = nn.Sequential(nn.Linear(1536, 740),
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
        model.last_linear = fv_last_linear
        
        if(dbg_run_deep):
            print("dbg_run_deep:")
            print(model)
            print("input_size", model.input_size)
        
     
        criterion = nn.NLLLoss()
        #optimizer = optim.Adam(model.last_linear.parameters(), lr=0.03)
        #optimizer = torch.optim.SGD(model.last_linear.parameters(),lr=0.05, momentum=0.9,  weight_decay=1e-4)
        optimizer = optim.SGD(model.parameters(), lr=0.10, momentum=0.9)
        sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 35], gamma=0.2, last_epoch=-1
        )
        
        
        MAX_steps = int(1e10)
        epochs = 100
        steps = 0
        running_loss = 0
        print_every = 20
        train_losses, test_losses = [], []
        valid_losses = []
        best_acc =0
        best_f1  = 0
        valid_loss =0
        test_loss = 0
        test_accuracy = 0
        valid_accuracy = 0
        train_accuracy = 0
        
        #model.summary()
       
        print("num classes = ", num_classes)
        if os.path.isfile("./save/best.pth"):
            print("Going to load saved best.pth")
            #model.load_state_dict(torch.load("./save/best.pth"))
            model= torch.load("./save/best.pth")
        else:
            print("No best.pth found")
            print("Starting from scratch")

        model.to(device)

        #begin add training
        with open(save_dir + "log.csv", "w") as myfile:
            myfile.write(
                "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % ("iter", "lr", "running_loss", "valid_loss", " test_loss", " train_accuracy", " valid_accuracy", " test_accuracy","valid f1", "test f1", "valid precision", "test precision", "valid recall", "test recall"))
            myfile.close()
        with open(save_dir + "Best_log.csv", "w") as myfile:
            myfile.write( "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % ("iter", "lr", "running_loss", "valid_loss", " test_loss", " train_accuracy", " valid_accuracy", " test_accuracy","valid f1", "test f1", "valid precision", "test precision", "valid recall", "test recall"))
            myfile.close()

        for epoch in range(epochs):
           # for batch in trainloader:
           #     print(batch[0].size())

            if(dbg_run):
                print("Epoch = ", epoch)
                #print("Len trainloader = ", len(trainloader))
                #print("Len test loader = ", len(testloader))
                #print("Len validloader = ", len(validloader))
            Run_computation = True;
            Check_Train_Accuracy = True;

            if(Run_computation):
                for inputs, labels in trainloader:

                    steps += 1
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    logps = model.forward(inputs)
                    if (dbg_run_deep):
                        print("dbg_run_deep: train shape logps =  ",np.shape(logps))
                    loss = criterion(logps, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    if (dbg_run):
                        print("train: ",steps)
                    train_accuracy=0
                    if steps % print_every == 0:

                        if (Check_Train_Accuracy):
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


                        test_loss = 0
                        test_accuracy = 0
                        test_f1=0
                        test_precision=0
                        test_recall=0
                        Test_Conf_Mat = np.zeros((num_classes, num_classes))
                        Valid_Conf_Mat = np.zeros((num_classes, num_classes))
                        valid_loss = 0
                        valid_accuracy = 0
                        valid_f1=0
                        valid_precision=0
                        valid_recall=0

                        model.eval()
                        with torch.no_grad():
                            #test accuracy
                            test_steps =0
                            for inputs, labels in testloader:

                                test_steps = test_steps+1
                                inputs, labels = inputs.to(device), labels.to(device)
                                logps = model.forward(inputs)
                                if (dbg_run_deep):
                                    print("dbg_run_deep: shape logps =  ",np.shape(logps))
                                batch_loss = criterion(logps, labels)
                                test_loss += batch_loss.item()

                                ps = torch.exp(logps)
                                if (dbg_run_deep):
                                    print("dbg_run_deep: shape ps =  ",np.shape(ps))
                                top_p, top_class = ps.topk(1, dim=1)
                                equals = top_class == labels.view(*top_class.shape)
                                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                                l_predict = []
                                my_labels = []

                                for x in top_class:
                                    for y in x:
                                       
                                        l_predict.append(y.item()) # l_predict=[y for x in top_class for y in x]
                                for x in labels:
                                        my_labels.append(x.item())

                                if (dbg_run_deep):
                                    print("my_labels shape = ", np.shape(my_labels));
                                    print("l_predict shape = ", np.shape(l_predict));
                                    
                                test_f1 += f1_score(my_labels, l_predict, average='micro')
                                test_precision += precision_score(my_labels, l_predict, average='micro')
                                test_recall += recall_score(my_labels, l_predict, average='micro')
                                Test_Conf_Mat += confusion_matrix(my_labels, l_predict, labels=range(num_classes))

                                
                                print("test_steps = ", test_steps)
                                    #Conf_Mat = confusion_matrix(my_labels, l_predict, labels=[0, 1, 2, 3, 4, 5, 6, 7])
                                    #print("test Conf_Mat (", len(testloader),")")
                                    #print("labels = (", len(labels), ")")
                                    #print("len test load = (", len(testloader), ")")
                            print(Test_Conf_Mat)
                            #file_name = "./save/Test_Conf_"+ time.strftime("%H%M%S")+".csv"
                            file_name = "./save/Test_Conf_"+ str(steps)+ "_" + time.strftime("%m%d")+"_" + time.strftime("%H%M%S")+".csv"
                           # np.savetxt(file_name, Test_Conf_Mat, delimiter=',', fmt='%1.0')
                            np.savetxt(file_name, Test_Conf_Mat,  delimiter=";", fmt='%4.0f')


                            #valid accuracy
                            for inputs, labels in validloader:
                                inputs, labels = inputs.to(device), labels.to(device)
                                logps = model.forward(inputs)
                                batch_loss = criterion(logps, labels)
                                valid_loss += batch_loss.item()

                                ps = torch.exp(logps)
                                top_p, top_class = ps.topk(1, dim=1)
                                equals = top_class == labels.view(*top_class.shape)
                                valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                                #flattening top_class
                                l_predict = []
                                my_labels = []
                                for x in top_class:
                                    for y in x:
                                        l_predict.append(y.item())
                                #l_predict=[y for x in top_class for y in x]
                                for x in labels:
                                    my_labels.append(x.item())
                                #l_predict=[y for x in top_class for y in x]
                                if (dbg_valid_deep):
                                    print("Valid :my_labels shape = ", np.shape(my_labels));
                                    print("Valid :l_predict shape = ", np.shape(l_predict));
                                    print("my_labels shape = ", np.shape(my_labels));
                                    print("l_predict shape = ", np.shape(l_predict));

                                valid_f1 += f1_score(my_labels, l_predict, average='micro')

                                if(save_Valid_conf):
                                    #valid_f1 += f1_score(my_labels, l_predict, average='micro')
                                    valid_precision += precision_score(my_labels, l_predict, average='micro')
                                    valid_recall += recall_score(my_labels, l_predict, average='micro')

                                    Valid_Conf_Mat += confusion_matrix(my_labels, l_predict, labels=range(num_classes))
                                    file_name = "./save/Valid_Conf_" + str(steps)+ "_" +  time.strftime("%m%d")+ "_" + time.strftime("%H%M%S") + ".csv"
                                    np.savetxt(file_name, Valid_Conf_Mat, delimiter=";" ,  fmt="%4.0f")
                            if(dbg_run_deep):
                                print("my labels = ", my_labels)
                                print("l_predict = ", l_predict)
                                                            
                            print("labels =( ", len(labels), ")")
                            print("valid acc = ", valid_accuracy / len(validloader))
                            print("Validation Confusion Matrix")
                            print(Valid_Conf_Mat)

                            #with open(save_dir + "log.csv", "a") as myfile:
                             #   myfile.write("%f,%f,%f;%f,%f,%f;%f,%f;%f,%f;%f,%f\n" % (
                             #   running_loss/ print_every, valid_loss/len(validloader), test_loss /len(testloader), train_accuracy/ print_every, valid_accuracy/ len(validloader), test_accuracy/len(testloader), valid_f1/ len(validloader), test_f1/len(testloader), valid_precision/ len(validloader), test_precision/len(testloader), valid_recall/ len(validloader), test_recall/len(testloader)))
                             #   myfile.close()

                            #if( valid_accuracy > best_acc):
                             #   print("saving best model ", str(valid_accuracy))
                             #   best_acc = valid_accuracy
                             #   torch.save(model, save_dir + "/best.pth")
                             #   with open(save_dir + "Best_log.csv", "a") as myfile:
                             #       myfile.write("%d %f,%f,%f;%f,%f,%f;%f,%f;%f,%f;%f,%f\n" % (steps, running_loss/ print_every, valid_loss/len(validloader), test_loss /len(testloader), train_accuracy/ print_every, valid_accuracy/ len(validloader), test_accuracy/len(testloader), valid_f1/ len(validloader), test_f1/len(testloader), valid_precision/ len(validloader), test_precision/len(testloader), valid_recall/ len(validloader), test_recall/len(testloader)))
                              #      myfile.close()

                            with open(save_dir + "log.csv", "a") as myfile:
                                print(optimizer.state_dict()["param_groups"][0]["lr"])
                                curr_lr = optimizer.state_dict()["param_groups"][0]["lr"]

                                myfile.write("%d, %f,  %f,%f,%f;%f,%f,%f;%f,%f;%f,%f;%f,%f\n" % (
                                steps, curr_lr, running_loss/ print_every, valid_loss/len(validloader), test_loss /len(testloader), train_accuracy/ print_every, valid_accuracy/ len(validloader), test_accuracy/len(testloader), valid_f1/ len(validloader), test_f1/len(testloader), valid_precision/ len(validloader), test_precision/len(testloader), valid_recall/ len(validloader), test_recall/len(testloader)))
                                myfile.close()

                            #if( valid_accuracy > best_acc):
                            if(valid_f1 > best_f1):
                                print("saving best model ", str(valid_accuracy))
                                print("saving best model ", str(valid_f1))
                                best_acc = valid_accuracy
                                best_f1 = valid_f1
                                torch.save(model, save_dir + "/best.pth")
                                
                                #evaluate precision recall
                                valid_precision += precision_score(my_labels, l_predict, average='micro')
                                valid_recall += recall_score(my_labels, l_predict, average='micro')

                                Valid_Conf_Mat += confusion_matrix(my_labels, l_predict, labels=range(num_classes))
                                file_name = "./save/Best_Valid_Conf_" + str(steps)+ "_" +  time.strftime("%m%d")+ "_" + time.strftime("%H%M%S") + ".csv"
                                np.savetxt(file_name, Valid_Conf_Mat, delimiter=";" ,  fmt="%4.0f")
                                
                                with open(save_dir + "Best_log.csv", "a") as myfile:
                                    myfile.write("%d, %f,  %f,%f,%f;%f,%f,%f;%f,%f;%f,%f;%f,%f\n" % (steps, curr_lr, running_loss/ print_every, valid_loss/len(validloader), test_loss /len(testloader), train_accuracy/ print_every, valid_accuracy/ len(validloader), test_accuracy/len(testloader), valid_f1/ len(validloader), test_f1/len(testloader), valid_precision/ len(validloader), test_precision/len(testloader), valid_recall/ len(validloader), test_recall/len(testloader)))
                                    myfile.close()
                                
                                #with open(save_dir + "Valid_Conf_Mat.csv", "w") as myfile:
                                #   myfile.write(Conf_Mat)
                                #   myfile.close()
                        train_losses.append(running_loss / print_every)
                        test_losses.append(test_loss / len(testloader))
                        valid_losses.append(valid_loss / len(validloader))
                        curr_lr = optimizer.state_dict()["param_groups"][0]["lr"]

                        print(f"Epoch {epoch + 1}/{epochs}.. "
                              f"Lr :{curr_lr:.3f}"
                              f"Train loss: {running_loss / print_every:.3f}.. "
                              f"Test loss: {test_loss / len(testloader):.3f}.. "
                              f"Train acc.: {train_accuracy/ print_every :.3f}.."
                              f"Test acc.: {test_accuracy / len(testloader):.3f}.."
                              f"Valid acc.: {valid_accuracy / len(validloader):.3f}.."
                              f"Test prec.: {test_precision / len(testloader):.3f}.."
                              f"Valid prec.: {valid_precision / len(validloader):.3f}.."
                              f"Test rec.: {test_recall / len(testloader):.3f}.."
                              f"Valid rec.: {valid_recall / len(validloader):.3f}"
                              )
                        running_loss = 0
                        model.train()
                        optimizer.step()  
                        sched.step()
        #end add training
        print("End training")

        sys.exit(0)

        model.eval()

        path_img = args.path_img
        # Load and Transform one input image
        load_img = utils.LoadImage()
        tf_img = utils.TransformImage(model)

        input_data = load_img(args.path_img) # 3x400x225
        input_data = tf_img(input_data)      # 3x299x299
        input_data = input_data.unsqueeze(0) # 1x3x299x299
        input = torch.autograd.Variable(input_data)

        # Load Imagenet Synsets
        with open('data/imagenet_synsets.txt', 'r') as f:
            synsets = f.readlines()

        # len(synsets)==1001
        # sysnets[0] == background
        synsets = [x.strip() for x in synsets]
        splits = [line.split(' ') for line in synsets]
        key_to_classname = {spl[0]:' '.join(spl[1:]) for spl in splits}

        with open('data/imagenet_classes.txt', 'r') as f:
            class_id_to_key = f.readlines()

        class_id_to_key = [x.strip() for x in class_id_to_key]

        # Make predictions
        output = model(input) # size(1, 1000)
        max, argmax = output.data.squeeze().max(0)
        print ("max = ",max)
        #print (argmax.shape())
        print (argmax)
        #class_id = argmax[0]
        class_id = argmax
        class_key = class_id_to_key[class_id]
        classname = key_to_classname[class_key]

        print("'{}': '{}' is a '{}'".format(arch, path_img, classname))

if __name__ == '__main__':
    main()
