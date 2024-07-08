#external packages
import random
import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#internal packages
from load_data import analyse_logs
from nn_structure import Data, NeuralNetwork, save_model, weighted_mse_loss, run_NN
from torch import nn

####################################

torch.random.manual_seed(77777)

#Directories
QMdata = "/share/workspace/eva/ML_stuff/SymmData/C6" #Data file
workdir = '/share/workspace/eva/ML_stuff/workdir' #Working directory with all the scripts

loss_data = 'loss.dat'
N = 6 #Number of ring atoms

#Load input (bl, ba, dh), and output (E, Fx, Fy, Fz), from QM data
Connectivity, Energy, Fx, Fy, Fz, Bonds, Angles, Dihedrals = analyse_logs(QMdata_file=QMdata, work_file=workdir, N=N)
NormE=np.reshape(Energy,(Energy.shape[0],1))

#Stack data, build input (X) and output (Y), and split
Y_data = np.hstack((NormE, np.array(Fx), np.array(Fy), np.array(Fz)))
X_data = np.hstack((Bonds, Angles, Dihedrals))
X_train, X_test, Y_train, Y_test = train_test_split(X_data,Y_data, test_size=0.3, shuffle=True, random_state=10)
print("Data split")

#Batch, format, and load the data
batch_size = 30
train_data = Data(X_train, Y_train)
test_data = Data(X_test, Y_test)

train_dataloader = DataLoader(dataset=train_data,
                                     shuffle=True,
                                     batch_size=batch_size,
                                     drop_last=True)
test_dataloader = DataLoader(dataset=test_data,
                                     shuffle=True,
                                     batch_size=batch_size,
                                     drop_last=True)
print("Data loaded")
print("Fecthing model and model parameters...")
#Define model parameters
num_epochs = 1000
input_dim = int(X_data.shape[1]) #Dimension of input X
hidden_dim = 60 #Size of hidden layers
output_dim = int(Y_data.shape[1]) #Dimension of output Y

#Define the remaining run parameters
epoch_cut='500' #Model is converged after the epoch cutoff and the eveluation of R2 can start
r2_cut='0.99' #Minimum R2 value at the training set
RegressionPlot=True
LearningPlot=True

print("Running NN...")

#Run training and test
run_NN(num_epochs, batch_size,input_dim, hidden_dim, output_dim, train_dataloader, test_dataloader, NormE, np.array(Fx), np.array(Fy), np.array(Fz),
        rsq_cut=r2_cut, epoch_cut=epoch_cut, RegressionPlot=RegressionPlot, LearningPlot=LearningPlot)

