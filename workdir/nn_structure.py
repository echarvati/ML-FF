import torch
import torch. nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import pandas as pd
from postprocess import plot_learning, plt_regression

class Data(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim, bias=False)

        torch.nn.init.trunc_normal_(self.layer_1.weight, mean=0.0, std=1.0, a=- 2.0, b=2.0)


        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.trunc_normal_(self.layer_2.weight, mean=0.0, std=1.0, a=- 2.0, b=2.0)


        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.trunc_normal_(self.layer_3.weight, mean=0.0, std=1.0, a=- 2.0, b=2.0)

        self.layer_4 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.trunc_normal_(self.layer_4.weight, mean=0.0, std=1.0, a=- 2.0, b=2.0)

        self.layer_5 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))

        x = torch.relu(self.layer_3(x))

        x = torch.sigmoid(self.layer_4(x))

        x = self.layer_5(x)

        return x

def save_model(model, optim, epoch, path):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict()}, path)

def weighted_mse_loss(input, target, weight):

    return (weight * (input - target) ** 2).sum() / weight.sum()


def run_NN(num_epochs, batch_size,input_dim, hidden_dim, output_dim, train_dataloader, test_dataloader, NormE, Fx, Fy, Fz,
           rsq_cut='rsq', epoch_cut='epoch_num',RegressionPlot=True, LearningPlot=True):



    model = NeuralNetwork(input_dim, hidden_dim, output_dim)

    # Optimizer
    learning_rate = 10 ** (-3)
    betas = (0.3, 0.93)
    epsilon = 10 ** (-20)
    decay = 10 ** (-8)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=betas, eps=epsilon, weight_decay=decay)

    loss_values = []
    log_train = []
    log_test = []

    for epoch in range(num_epochs):
        for X, y in train_dataloader:
            X = X.type(torch.float)
            X = X.view(X.size(0), -1)
            y = y.type(torch.float)
            y = y.view(X.size(0), -1)
            y_pred = model(X)

            # Define weights
            weight_E = np.full((int(y.shape[0]), (int(NormE.shape[1]))), 1.0)
            weight_Gradx = np.full((int(y.shape[0]), (int(np.array(Fx).shape[1]))), 0.1)
            weight_Grady = np.full((int(y.shape[0]), (int(np.array(Fy).shape[1]))), 0.1)
            weight_Gradz = np.full((int(y.shape[0]), (int(np.array(Fz).shape[1]))), 0.1)
            weights = torch.FloatTensor(np.hstack((weight_E, weight_Gradx, weight_Grady, weight_Gradz)))

            step_loss = weighted_mse_loss(y_pred, y, weight=weights)
            loss_values.append(step_loss.item())

            # Backward pass: derivatives of loss function
            optimizer.zero_grad()
            step_loss.backward()

            # Update regression parameters
            optimizer.step()

        r2 = r2_score(y_true=y.detach().flatten().tolist(), y_pred=y_pred.detach().flatten().tolist())
        mae = mean_absolute_error(y_true=y.detach().flatten().tolist(), y_pred=y_pred.detach().flatten().tolist())
        log_train.append([epoch, r2, mae])
        if r2 >= float(rsq_cut) and epoch > int(epoch_cut):
            torch.save(model.state_dict(), 'model_weights.pth')
            print('METRICS: epoch Rsq MAE', epoch, r2, mae)

    # Check if the model was learning during training.
    # This is useful for hyperparamter optimization.
    if LearningPlot == True:
        plot_learning(num_epochs, loss_values, batch_size, learning_rate)


    device = 'cpu'
    print('Batched Training complete')

    model.load_state_dict(torch.load('model_weights.pth', map_location=device))
    model.eval()
    y_all = []
    y_pred_all = []
    for X, y in train_dataloader:
        X = X.type(torch.float)
        X = X.view(X.size(0), -1)
        y = y.type(torch.float)
        y = y.view(y.size(0), -1)
        y_pred = model(X)
        y_pred_dat = y_pred.detach().flatten().tolist()
        y_data = y.detach().flatten().tolist()
        y_all = y_all + y_data
        y_pred_all = y_pred_all + y_pred_dat

    r2 = r2_score(y_true=y_all, y_pred=y_pred_all)
    mae = mean_absolute_error(y_true=y_all, y_pred=y_pred_all)
    train_dia = {'real': y_all, 'pred': y_pred_all}
    print('Overall for the training set [Rsq MAE]:', r2, mae)

    y_all_T = []
    y_pred_all_T = []

    for X, y in test_dataloader:
        X = X.type(torch.float)
        X = X.view(X.size(0), -1)
        y = y.type(torch.float)
        y = y.view(y.size(0), -1)
        y_pred = model(X)
        y_pred_dat = y_pred.detach().flatten().tolist()
        y_data = y.detach().flatten().tolist()
        y_all_T = y_all_T + y_data
        y_pred_all_T = y_pred_all_T + y_pred_dat
    test_dia = {'real': y_all_T, 'pred': y_pred_all_T}
    test_frame = pd.DataFrame(test_dia)
    test_frame.to_csv('test_set.csv')
    r2_test = r2_score(y_true=y_all_T, y_pred=y_pred_all_T)
    mae_test = mean_absolute_error(y_true=y_all_T, y_pred=y_pred_all_T)
    print('Performance at the test set [Rsq MAE]:', r2_test, mae_test)

    if RegressionPlot==True:
        plt_regression(y_all_T, y_pred_all_T, stage='Test', title='title')