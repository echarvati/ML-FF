import matplotlib.pyplot as plt
import numpy as np

def plot_learning(num_epochs, loss_values, batch_size,learning_rate):

    step = np.linspace(0, num_epochs, len(loss_values))
    fig1, ax = plt.subplots(figsize=(8, 5))
    plt.plot(step, np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    fig1.savefig('LearningCurve_%i_%i_%s.jpg' % (num_epochs, batch_size, str(learning_rate)), bbox_inches='tight')


def plt_regression(y_true, y_pred, stage ='stage',title = 'title'):

    fig1, ax = plt.subplots(figsize=(8, 5))
    plt.title("Regression results at the %s set" %stage)
    plt.plot(y_true, y_true, color='black')
    plt.scatter(y_true, np.array(y_pred), color='orange')
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    fig1.savefig('RegressionPlot_%s.jpg' %stage, bbox_inches='tight')
    plt.close()


