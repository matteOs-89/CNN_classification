
# DOWNLOAD LIBRARY

import numpy as np
import torch
import torch.nn as nn
import  torch.nn.functional as F

from torchsummary import summary
import torchvision.transforms as T
import torchvision

import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt


"""
 EMNIST CLASSIFICATION

The aim of this project is to Build a Convolutional neural network model that can successfully classify the EMNIST Dataset which is available in pytorch dataset available online.
we will carry out some visual analysis of the model performance and perhaps suggest how the model inference could be improve, and reasons while its may have some missclassifications.
some of the letters.

"""

# Download the dataset from torchvision database

EmnistData = torchvision.datasets.EMNIST(root="emnist", 
                                    split="letters", 
                                    download=True)


EmnistData.class_to_idx
classindex = EmnistData.classes
print(classindex)

# checking class to index and noticing 0 has no letter assigned to it, therefore it is not needed.

classindex = EmnistData.classes[1:]
print(classindex)

EmnistData.targets.sum()

print(EmnistData.data.shape)

train_images = EmnistData.data.view(124800, 1, 28,28).float()

print(train_images.shape)

labels = EmnistData.targets
print(torch.sum(labels==0))


labels = (EmnistData.targets)-1 #Removing the extra target
print(labels.shape)

print(torch.sum(labels==0))

train_images/= torch.max(train_images) # Normalization

print(train_images.max())
print(train_images.min())




def visualizeImage(train_images, classindex,labels):
  """
  This function enables us the ability to visualize
  randomly selected images from our dataset and its true label
  """

  fig, ax = plt.subplots(1,2, figsize=(6,4))

  for (i, ax) in enumerate(ax.flatten()):

    randomIMG = np.random.randint(train_images.shape[0]) 

    i = np.squeeze(train_images[randomIMG,:,:])
    targetL = classindex[labels[randomIMG]]

    ax.imshow(i.T, cmap="gray")
    ax.set_title(f"the letter {targetL}'")

    plt.show()


visualizeImage(train_images, classindex,labels)






X_train, X_test, y_train, y_test = train_test_split(train_images, labels, test_size=0.2)
Val_data, X_test, Val_label, y_test = train_test_split(X_test, y_test, test_size=0.5)


train_data = TensorDataset(X_train, y_train)

val_data = TensorDataset(Val_data, Val_label)

test_data = TensorDataset(X_test, y_test)


batchsize = 64
train_loader = DataLoader(train_data,
                          shuffle=True,
                          batch_size=batchsize,
                          drop_last=True)


val_loader = DataLoader(val_data,
                          shuffle=False,
                          batch_size=test_data.tensors[0].shape[0],
                          )

test_loader = DataLoader(test_data,
                          shuffle=False,
                          batch_size=test_data.tensors[0].shape[0],
                          )


print(train_loader.dataset.tensors[0].shape[0])
print(val_loader.dataset.tensors[0].shape[0])   
print(test_loader.dataset.tensors[0].shape[0])




def CreatetheNetwork():

  """
  This function builds the Convolutional neural network architectures, and also establish
  the loss function and optimizer used for this experiment.
  """

  class emnistnet(nn.Module):

    def __init__(self, check_shape):
      super().__init__()

      self.conv1 = nn.Conv2d(1,64,3, padding=1)
      self.conv2 = nn.Conv2d(64, 128, 3, padding =1)
      self.conv3 = nn.Conv2d(128,128,3, padding=1)
      self.conv4 = nn.Conv2d(128, 256, 3, padding =1)


      self.bn1 = nn.BatchNorm2d(64)
      self.bn2 = nn.BatchNorm2d(128)
      self.bn3 = nn.BatchNorm2d(128)
      self.bn4 = nn.BatchNorm2d(256)


      self.fc1 = nn.Linear(1*1*256, 256)
      self.fc2 = nn.Linear(256, 128)
      self.fc3 = nn.Linear(128, 26)


    def forward(self, x):
      


      x = F.max_pool2d(self.conv1(x), 2)
      x = F.leaky_relu(self.bn1(x))
      x = F.dropout(x, p=0.2, training=True)

 


      x = F.max_pool2d(self.conv2(x), 2)
      x = F.leaky_relu(self.bn2(x))
      x = F.dropout(x, p=0.2, training=True)



      x = F.max_pool2d(self.conv3(x), 2)
      x = F.leaky_relu(self.bn3(x))
      x = F.dropout(x, p=0.2, training=True)


      x = F.max_pool2d(self.conv4(x), 2)
      x = F.leaky_relu(self.bn4(x))

      

      nunits =x.shape.numel()/x.shape[0]
      x = x.view(-1,int(nunits))



      x = F.leaky_relu(self.fc1(x))
      x = F.dropout(x, p=0.5, training=True)

      

      x = F.leaky_relu(self.fc2(x))
      x = F.dropout(x, p=0.5, training=True)
  
      
 
      x = self.fc3(x)


      return x


  net = emnistnet()

  lossfun = nn.CrossEntropyLoss()

  optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

  return net, lossfun, optimizer


net,lossfun,optimizer = CreatetheNetwork()



checkpoint = {"state_dict": net.state_dict(), 
              "optimizer": optimizer.state_dict(),
              "loss" : 0.2
                 }


def store_checkpoint(state, filename="my-checkpoint.pth.tar"):
    print("saving Checkpoint...")
    torch.save(state, filename)



def load_checkpoint(checkpoint):
  print("loading checkpoint")
  checkpoint= torch.load("my-checkpoint.pth.tar")
  net.load_state_dict(checkpoint["state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer"])



def trainCreatedModel(net, lossfun=lossfun, optimizer=optimizer, 
                      train_loader=train_loader, val_loader=test_loader, 
                      numepochs=10, load_model=True ):

  """
  This function train and evaluate the model while also saving specified checkpoints specified.
  """

  Trainloss= torch.zeros(numepochs)
  Testloss = torch.zeros(numepochs)
  TrainACC = torch.zeros(numepochs)
  TestACC  = torch.zeros(numepochs)


  for epoch in range(numepochs):

    net.train()
    batchloss = []
    batchACC  = []

    if epoch % 5==0:
      
      store_checkpoint(checkpoint)


    for X,y in train_loader:

      
      yhat = net(X)
      loss = lossfun(yhat, y)

      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      
      batchloss.append(loss.item())
      batchACC.append(torch.mean((torch.argmax(yhat,axis=1) == y).float()).item() )

    Trainloss[epoch]=np.mean(batchloss)
    TrainACC[epoch] = 100*np.mean(batchACC)



    # Using validation dataset to validate the model performance
    net.eval()
    with torch.no_grad():

      X,y = next(iter(val_loader))


      T_pred = net(X)
      loss = lossfun(T_pred, y)

      Testloss[epoch] = loss
      TestACC[epoch] = 100*torch.mean((torch.argmax(T_pred, axis=1)==y).float()).item()
      
  

    
    msg = f"Finished epoch {epoch +1}/{numepochs} Train Accuracy: {batchACC[-1]} Train Loss: {batchloss[-1]}"
    sys.stdout.write("\r" + msg)
   

  return Trainloss, Testloss, TrainACC, TestACC

Trainloss, Testloss, TrainACC, TestACC = trainCreatedModel(net, lossfun=lossfun, optimizer=optimizer, 
                    train_loader=train_loader, val_loader=val_loader, 
                      numepochs=10, load_model=True)




"""
from the graph below, its shows the training data performed better than the evaluation data, however the margin is about 1% which is not bad.
There also was some spikes on the validation set during training as it seems like our model showed signs of inconsistence.
"""

fig,ax = plt.subplots(1,2, figsize=(8,5))

ax[0].plot(Trainloss, "s-", label="Train")
ax[0].plot(Testloss, "s-", label="Test")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].set_title("Model loss")

ax[1].plot(TrainACC, "s-", label="Train")
ax[1].plot(TestACC, "s-", label="Test")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Error (%)")
ax[1].set_title(f"Final Model test ACC rate{TestACC[-1]:.2f}%")
ax[1].legend()

plt.show()


# Test the model on unseen data and evaluate its performance. 

net.eval()
with torch.no_grad():
   
   testAcc = []
   testloss = []
   X,y = next(iter(test_loader))


   Test_pred = net(X)
   loss = lossfun(Test_pred, y)

   testloss.append(loss)
   testAcc.append(100*torch.mean((torch.argmax(Test_pred, axis=1)==y).float()).item())
   


randindex = np.random.choice(len(y), size=21, replace=False)

fig, ax =plt.subplots(3,7, figsize=(8,4))

for i,ax in enumerate(ax.flatten()):


  I = np.squeeze(X[randindex[i],0,:,:])
  trueletter = classindex[ y[randindex[i]] ]
  predletter = classindex[torch.argmax(Test_pred[randindex[i],:])]


  col = "gray" if trueletter == predletter else "hot"

  

  ax.imshow(I.T, cmap=col)
  ax.set_title("Time %s, predict %s" %(trueletter, predletter), fontsize=6)
  ax.set_xticks([])
  ax.set_yticks([])

plt.show()



# Compute the confusion matrix
Plot_C = confusion_matrix(y,torch.argmax(Test_pred,axis=1),normalize='true')


fig = plt.figure(figsize=(8,8))
plt.imshow(Plot_C,'Blues',vmax=.03)

plt.xticks(range(26),labels=classindex)
plt.yticks(range(26),labels=classindex)
plt.title(f'TEST confusion matrix {testAcc[-1]}')
plt.xlabel('Predicted  Label')

plt.show()

"""From the confusion matrix plot we can see that the computer got alot of predictions right, however we can also see that or model made some distinct errors,
Some of the errors with strong correlations are "predicting Q instead A, predicting G instead Q, predicting H instead N, predicting I instead L.

Some methods that which could improve the performs include:

- Data Augmentation
- Prepocessing technique to further removing Noise
- Checking Dataset to ensure or data are rightly labeled
- Update model architecture
- Try weight initialization
- Impliment regulization techniques
"""


