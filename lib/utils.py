from lib.ibelief import decisionDST
import matplotlib.pyplot as plt
import numpy as np
import random

# Transform evidential labels into classes
def mass_to_labels(y):
    y_labels = np.zeros(y.shape[0])

    for i in range(y.shape[0]):
        # Using Pignistic probabilities
        y_labels[i] = decisionDST(y[i].T, 4)
    return y_labels

# Plot results
def plot(pred, start):
    pred = np.array(pred)
    requests = [i + 1 + start for i in range(pred.shape[2])]

    plt.title("Accuracy by number of labeled instances")
    plt.plot(requests, np.mean(pred[:,1,:], axis=0), '-', label='$\gamma_i$-EKNN')
    plt.plot(requests, np.mean(pred[:,2,:], axis=0), ':',label='Logistic Regression')
    plt.plot(requests, np.mean(pred[:,0,:], axis=0), '--',label='K-NN')
    plt.plot(requests, np.mean(pred[:,3,:], axis=0), '-.',label='Random forest')
    plt.xticks([i * 2 + start for i in range(int(pred.shape[2] / 2) + 1)])
    plt.plot(label='Sample Label Blue')
    plt.xlabel("Number of labeled instances")
    plt.ylabel("Mean accuracy")
    plt.legend()
    plt.show()

# Plot results
def plot_min_samples(pred, min_samples_leaf):
    pred = np.array(pred)
    min_samples_leaf = np.array(min_samples_leaf)

    plt.title("Accuracy by number of min samples in each leaf")
    plt.plot(min_samples_leaf, pred[0,:], label='DT')
    plt.plot(min_samples_leaf, pred[1,:], label='Euclidean-EDT')
    plt.plot(min_samples_leaf, pred[2,:], label='Uncertainty-EDT')
    plt.plot(min_samples_leaf, pred[3,:], label='Jousselme-EDT')
    plt.plot(min_samples_leaf, pred[4,:], label='Conflict-EDT')

    plt.plot(label='Sample Label Blue')
    plt.xlabel("Number of min samples in leaf")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# Plot results
def plot_noise(pred, noise):
    pred = np.array(pred)
    noise = np.array(noise)

    plt.title("Accuracy by noise")
    plt.plot(noise, pred[0,:],':', label='DT')
    plt.plot(noise, pred[1,:], label='Euclidean-EDT')
    plt.plot(noise, pred[2,:], label='Uncertainty-EDT')
    plt.plot(noise, pred[3,:], label='Jousselme-EDT')
    plt.plot(noise, pred[4,:],'--', label='Conflict-EDT')

    plt.plot(label='Sample Label Blue')
    plt.xlabel("Proportion of noise")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower left')
    plt.show()

# Plot results
def plot_lbda(pred, noise):
    pred = np.array(pred)
    noise = np.array(noise)

    plt.title("Accuracy by $\lambda$ value")
    plt.plot(noise, pred, label='Uncertainty-EDT')

    plt.plot(label='Sample Label Blue')
    plt.xlabel("Value of $\lambda$")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def plot_lambdas(pred, lbda):
    pred = np.array(pred)
    lbda = np.array(lbda)

    plt.title("Accuracy by lambda")
    plt.plot(lbda, pred, label='Uncertainty-EDT')
    #plt.xticks([i * 2 + start for i in range(int(pred.shape[2] / 2) + 1)])
    plt.plot(label='Sample Label Blue')
    plt.xlabel("Lambda value")
    plt.ylabel("Mean accuracy")
    plt.legend()
    plt.show()

# Plot results
def plot_al(pred, start):
    pred = np.array(pred)
    requests = [i + 1 + start for i in range(pred.shape[2])]

    plt.title("Accuracy by number of labeled instances")
    plt.plot(requests, np.mean(pred[:,1,:], axis=0), label='Uncertainty')
    #plt.fill_between(requests, np.mean(pred[:,1,:], axis=0) - (1.96 * np.std(pred[:,1,:], axis=0)) / np.sqrt(pred[:,1,:].shape[0]), np.mean(pred[:,1,:], axis=0) + (1.96 * np.std(pred[:,1,:], axis=0)) / np.sqrt(pred[:,1,:].shape[0]), color='b', alpha=.1)

    #plt.plot(requests, np.mean(pred[:,2,:], axis=0), ':',label='Shannon entropy')

    #plt.plot(requests, np.mean(pred[:,0,:], axis=0),label='Random')

    #plt.plot(requests, np.mean(pred[:,3,:], axis=0), '-.',label='Non-specificity')

    #plt.plot(requests, np.mean(pred[:,4,:], axis=0), label='Evidential Uncertainty')

    #plt.plot(requests, np.mean(pred[:,5,:], axis=0), '-.',label='Discord')

    #plt.plot(requests, np.mean(pred[:,6,:], axis=0), '--',label='Evidential Uncertainty 2')

    plt.plot(requests, np.mean(pred[:,7,:], axis=0), ':', label='EDT + Uncertainty')

    #plt.plot(requests, np.mean(pred[:,8,:], axis=0),label='Committee')

    #plt.plot(requests, np.mean(pred[:,9,:], axis=0),label='Evidential Committee')

    plt.xticks([i * 2 + start for i in range(int(pred.shape[2] / 2) + 1)])
    plt.plot(label='Sample Label Blue')
    plt.xlabel("Number of labeled instances")
    plt.ylabel("Mean accuracy")
    plt.legend()
    plt.show()

# Plot results normalized
def plot_al_norm(pred, start):
    pred = np.array(pred)
    requests = [i + 1 + start for i in range(pred.shape[2])]

    plt.title("Accuracy by number of labeled instances")
    plt.plot(requests, np.mean(pred[:,3,:], axis=0), label='EK-NN Klir $\lambda$ = 0.5')
    plt.plot(requests, np.mean(pred[:,4,:], axis=0), label='EK-NN Klir $\lambda$ = 0.3')
    plt.plot(requests, np.mean(pred[:,5,:], axis=0), label='EK-NN Klir $\lambda$ = 0.7')
    plt.plot(requests, np.mean(pred[:,0,:], axis=0), label='EK-NN Klir Norm $\lambda$ = 0.5')
    plt.plot(requests, np.mean(pred[:,1,:], axis=0), label='EK-NN Klir Norm $\lambda$ = 0.3')
    plt.plot(requests, np.mean(pred[:,2,:], axis=0), label='EK-NN Klir Norm $\lambda$ = 0.7')


    plt.xticks([i * 2 + start for i in range(int(pred.shape[2] / 2) + 1)])
    plt.plot(label='Sample Label Blue')
    plt.xlabel("Number of labeled instances")
    plt.ylabel("Mean accuracy")
    plt.legend()
    plt.show()

def noise_imprecision(y, nb_classes, classes, noise=0):
    # Prepare random with reproductibility
    random.seed(974)

    # Noise on y and y_cred
    indices = np.array([i for i in range(0, y.shape[0])])
    y_cred = np.zeros((y.shape[0], 2**nb_classes))
    
    for i in range(0, y.shape[0]):
        y_cred[i][2**y[i]] = 1

    for i in range(0, int(noise * y.shape[0])):
        rand_indice = random.choice(indices) 
        new_label = random.choice(np.delete(classes, y[rand_indice]))

        # Noise on y_cred
        y_cred[rand_indice][2**y[rand_indice]] = 0
        y_cred[rand_indice][2**new_label + 2**y[rand_indice]] = 1

        # Noise on y
        y[rand_indice] = decisionDST(y_cred[rand_indice].T, 4)

        indices = np.delete(indices, np.where(indices == rand_indice))
    
    return y, y_cred

def noise_mistake(y, nb_classes, classes, noise=0):
        # Prepare random with reproductibility
    random.seed(974)

    # Noise on y and y_cred
    indices = np.array([i for i in range(0, y.shape[0])])
    y_cred = np.zeros((y.shape[0], 2**nb_classes))
    
    for i in range(0, y.shape[0]):
        y_cred[i][2**y[i]] = 1

    for i in range(0, int(noise * y.shape[0])):
        rand_indice = random.choice(indices) 
        new_label = random.choice(np.delete(classes, y[rand_indice]))

        # Noise on y_cred
        y_cred[rand_indice][2**y[rand_indice]] = 0
        y_cred[rand_indice][2**new_label] = 1

        # Noise on y
        y[rand_indice] = new_label

        indices = np.delete(indices, np.where(indices == rand_indice))
    
    return y, y_cred

    return 0

# Add noise to data
def noising(y, nb_classes, classes, noise=0):
    # Prepare random
    # Reproductibility
    random.seed(974)

    # Noise on y and y_cred
    indices = np.array([i for i in range(0, y.shape[0])])
    y_cred = np.zeros((y.shape[0], 2**nb_classes))
    
    for i in range(0, y.shape[0]):
        y_cred[i][2**y[i]] = 1

    for i in range(0, int(noise * y.shape[0])):
        rand_indice = random.choice(indices) 
        new_label = random.choice(np.delete(classes, y[rand_indice]))
        random_num = random.random()

        # Noise on y_cred
        y_cred[rand_indice][int(2**new_label)] = 1
        for n in range(1, 2**nb_classes):
            if(2**new_label == n):
                y_cred[rand_indice][n] = random_num
            else:
                y_cred[rand_indice][n] = (1 - random_num) / (2**nb_classes - 2)

        # Noise on y
        y[rand_indice] = decisionDST(y_cred[rand_indice].T, 4)

        indices = np.delete(indices, np.where(indices == rand_indice))
    
    return y, y_cred
