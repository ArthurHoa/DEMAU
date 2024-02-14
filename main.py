from os import walk
import webbrowser

import matplotlib.pyplot as plt
from matplotlib import cm as CM
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from scipy.optimize import minimize_scalar

import lib.PySimpleGUI as sg
from matplotlib.colors import ListedColormap
from lib.eknn_imperfect import EKNN
from lib.utils import noise_imprecision
from lib import ibelief

import pandas as pd
import numpy as np
import uncertainties
import models
import math

######################################################
# This main code runs the app                        #
#                                                    #
#                                                    #
# Modify model.py to append your own models          #
#                                                    #
# Modify uncertainties.py to append your own methods #
#                                                    #
#                                                    #
# Arthur H. 2024                                     #
# https://github.com/ArthurHoa                       #
#                                                    #
######################################################

# Possible rendering sizes for uncertainty space
SIZE = [20, 30, 40, 50, 60, 70]

# Datasets class colors
COLORS = [[0, .8, 0], [.8, 0, .8], [.8, .8, 0], [.8, 0, 0], [0, .8, .8], [0.5, 0.5, 0.5], [0, 0, .8]]

# Generate layout
def generateLayout(datasets_list, features):
    font_title = ("normal", 14)
    layout = [
        [   
            # Title
            sg.Text("DEMAU: Decompose, Explore, Model and Analyze Uncertainties", font=font_title)
        ],
        [   
            # The 2D figures: Dataset + Uncertainties
            sg.Pane([
                sg.Column([[sg.Canvas(key='DATASET', size=(500,400))]]),
                sg.Column([[sg.Canvas(key='UNCERTAINTIES', size=(500,400))]])], orientation='h', show_handle=False, border_width=3, relief=sg.RELIEF_SUNKEN, pad=(0,15))
        ],
        [
            # Settings options
            sg.Column([
                [sg.Text("Dataset")],
                [sg.Combo(datasets_list, default_value="Iris", s=(15,22), enable_events=True, readonly=True, k='DATASET_COMBO')],
                [sg.Text()],
                [sg.Text("Variables", key="VARIABLES")],
                [sg.Combo(list(features), default_value=features[0], s=(15,22), enable_events=True, readonly=True, k='VAR1_COMBO')],
                [sg.Combo(list(features), default_value=features[1], s=(15,22), enable_events=True, readonly=True, k='VAR2_COMBO')]], pad=(18,0)),
            sg.Column([
                [sg.Text("Model")],
                [sg.Combo(models.get_models(), default_value="SVM", s=(15,22), enable_events=True, readonly=True, k='MODELS_COMBO')],
                [sg.Text(pad=(0,15))],
                [sg.Text("Uncertainty")],
                [sg.Combo(uncertainties.get_uncertaintes(), default_value=uncertainties.get_uncertaintes()[0], s=(15,22), enable_events=True, readonly=True, k='UNCERTAINTY')]], pad=(18,0)),
            sg.Column([
                [sg.Text(pad=((0,0),(7,0)))],
                [sg.Text("Type of Uncertainty")],
                [sg.Text(pad=((0,0),(4,0)))],
                [sg.Radio("Total Uncertainty", 1, default="True", key="TOTAL", enable_events=True)],
                [sg.Radio("Aleatoric & Epistemic", 1, key="EPISTEMIC", enable_events=True)],
                [sg.Radio("Discord  & Non-Specificity", 1, key="EVIDENTIAL", enable_events=True)]], pad=(18,0)),
            sg.Column([
                [sg.Text(pad=((0,0),(7,0)))],
                [sg.Text("Dimensionality reduction")],
                [sg.Text(pad=((0,0),(0,2)))],
                [sg.Checkbox('PCA', key="PCA", enable_events=True)],
                [sg.Checkbox('SVD', key="SVD", enable_events=True)],
                [sg.Checkbox('TSNE', key="TSNE", enable_events=True)]], pad=(18,0)),
            sg.Column([
                [sg.Text(pad=((0,0),(0,0)))],
                [sg.Text("Grid size")],
                [sg.Checkbox('Auto', key="GRID", default=True, enable_events=True)],
                [sg.Slider((1,6), default_value=6, enable_events=True, orientation='h', s=(12,17), key='RENDER', visible=False)]], pad=(18,0))
        ],
        [   
            # Biblio
            sg.Text("", font=font_title),
            sg.Text("", font=font_title)
        ],
        [   
            # Biblio
            sg.Text("", key="BIBLIO"),
            sg.Text("")
        ],
        [
            # Extra buttons
            sg.Pane([
            sg.Column([[sg.Button("Refresh", key="REFRESH")]]),
            sg.Column([[sg.Button("Save", key="SAVE")]]),
            sg.Column([[sg.Button("Contacts", key="CONTACTS")]]),
            sg.Column([[sg.Button("Exit")]])], orientation='h', show_handle=False, border_width=1, relief=sg.RELIEF_SUNKEN, pad=((0,0),(20,10)))
        ]
    ]

    window = sg.Window("DEMAU", layout, finalize=True, location=(200, 150), element_justification='center', use_default_focus=False)

    return window

# Grid values
R3 = 1.73205
SIZES = [[int(SIZE[0] * R3), SIZE[0]], [int(SIZE[1] * R3), SIZE[1]], [int(SIZE[2] * R3), SIZE[2]], [int(SIZE[3] * R3), SIZE[3]], [int(SIZE[4] * R3), SIZE[4]], [int(SIZE[5] * R3), SIZE[5]]]

# Draw 2D plot and append to layout
def draw_figure(canvas, figure):
   figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
   figure_canvas_agg.draw()
   figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
   return figure_canvas_agg

# Modify 2D figure
def modify_figure(figure_canvas):
    figure_canvas.draw()
    figure_canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas

# Compute total uncertainty
def compute_uncertainty(X, y, X_test):
    classifier.fit(X, y)
    proba = classifier.predict_proba(X_test)
    return uncertainties.uncertainty(uncertainty, proba)

# Compute epistemic uncertainty
def compute_epistemic_uncertainty(X, y, X_test):

    neighbors = 7

    # Load model for classification
    classifier = models.load_model("K-NN")

    # Fit model according to the dataset
    classifier.fit(X, y)
    
    epistemic_list = []
    aleatoric_list = []

    # Compute epistemic uncertainty
    for i in range(X_test.shape[0]):
        dist, indices = classifier.kneighbors(np.array([X_test[i]]), neighbors)

        epistemic, aleatoric = compute_epistemic(dist[0], indices[0], np.array(y))
        epistemic_list.append(epistemic)
        aleatoric_list.append(aleatoric)

    return epistemic_list, aleatoric_list

# Compute discord
def compute_discord(X, y, X_test):

    # Format data for EK-NN
    _, y_cred = noise_imprecision(y, np.max(y) + 1, np.array(list(set(y))), 0)

    # Load and train EK-NN
    neighbors = 7
    classifier = EKNN(np.max(y) + 1, n_neighbors=neighbors)
    classifier.fit(X, y_cred, alpha=1, beta=2)
    
    discord_list = []

    # Compute discord
    for i in range(X_test.shape[0]):
        _, bbas = classifier.predict(np.array([X_test[i]]), return_bba=True)

        pign_prob = np.zeros((bbas.shape[0], bbas.shape[1]))
        for k in range(bbas.shape[0]): 
                betp_atoms = ibelief.decisionDST(bbas[k].T, 4, return_prob=True)[0]
                for i in range(1, bbas.shape[1]):
                    for j in range(betp_atoms.shape[0]):
                            if ((2**j) & i) == (2**j):
                                pign_prob[k][i] += betp_atoms[j]

                    if pign_prob[k][i] != 0:
                        pign_prob[k][i] = math.log2(pign_prob[k][i])

        discord_list.append(-np.sum(bbas[0] * pign_prob[0]))

    return discord_list

# Compute non-specificity
def compute_nonspe(X, y, X_test):

        # Format data for EK-NN
    _, y_cred = noise_imprecision(y, np.max(y) + 1, np.array(list(set(y))), 0)

    # Load and train EK-NN
    neighbors = 7
    classifier = EKNN(np.max(y) + 1, n_neighbors=neighbors)
    classifier.fit(X, y_cred, alpha=1, beta=2)
    
    nonspe_list = []

    # Compute non-specificity
    for i in range(X_test.shape[0]):
        _, bbas = classifier.predict(np.array([X_test[i]]), return_bba=True)

        card = np.zeros(bbas.shape[1])
        for i in range(1, bbas.shape[1]):
            card[i] = math.log2(bin(i).count("1"))

        nonspe_list.append(np.sum(bbas[0] * card))

    return nonspe_list

# Load dataset from CSV
def load_dataset(path):
    return pd.read_csv(path)

# Load the uncertainty grid
def load_Xtest(X, size):
    X_test = np.zeros((SIZES[size][0] * SIZES[size][1], 2))

    minx1 = np.min(X[:,0])
    minx1 = minx1 + max(minx1, -minx1) * 0.01
    maxx1 = np.max(X[:,0])
    maxx1 = maxx1 - max(maxx1, -maxx1) * 0.01
    minx2 = np.min(X[:,1])
    minx2 = minx2 + max(minx2, -minx2) * 0.01
    maxx2 = np.max(X[:,1])
    maxx2 = maxx2 - max(maxx2, -maxx2) * 0.01

    for i in range(SIZES[size][0]):
        for j in range(SIZES[size][1]):
            X_test[i + j * SIZES[size][0]][0] = minx1 + i * (maxx1 - minx1) / SIZES[size][0]
            X_test[i + j * SIZES[size][0]][1] = minx2 + j * (maxx2 - minx2) / SIZES[size][1]

    return X_test

# Compute epistemic uncertainty
def compute_epistemic(dist, indices, y):
    nb_classes = 2
    res = np.zeros(nb_classes)

    for i in range(indices.shape[0]):
        res[y[indices[i]]] += (1/dist[i])

    p = res[0]
    n = res[1]

    opt = minimize_scalar(f_objective_1, bounds=(0, 1), method='bounded', args=(p, n))
    pl1 = opt.x

    opt = minimize_scalar(f_objective_2, bounds=(0, 1), method='bounded', args=(p, n))
    pl2 = 1 - opt.x

    ue = min(pl1, pl2) - 0.5
    ua = 1 - max(pl1, pl2)

    return ue, ua

# Objective fuction used to compute epistemic uncertainty
def f_objective_1(theta, p, n):
    left = ((theta**p) * (1-theta)**n) / (((p / (n+p))**p) * ((n / (n+p))**n))
    right = 2 * theta - 1

    res = min(left, right)

    return -res

# Objective fuction used to compute epistemic uncertainty
def f_objective_2(theta, p, n):
    left = ((theta**p) * (1-theta)**n) / (((p / (n+p))**p) * ((n / (n+p))**n))
    right = 1 - (2 * theta)
        
    res = min(left, right)

    return -res

# Scatter plot dataset figure
def plot_dataset(X1, X2, y, fig = None):

    # Create or clear figure
    if fig == None:
        fig = plt.figure(figsize=(5, 4), dpi=100)
    else:
        fig.clear()

    subplot = fig.add_subplot(111)
    subplot.scatter(X1, X2, c=y, cmap=ListedColormap(COLORS))
    
    # Remove black rectangle
    subplot.spines['top'].set_visible(False)
    subplot.spines['right'].set_visible(False)
    subplot.spines['bottom'].set_visible(False)
    subplot.spines['left'].set_visible(False)

    # Remove ticks form figure
    subplot.set(yticklabels=[], xticklabels=[]) 
    subplot.tick_params(left=False, bottom=False) 

    # Same size for each figure
    subplot.set_xlim(np.min(X1) - 0.2, np.max(X1) + 0.2)
    subplot.set_ylim(np.min(X2) - 0.2, np.max(X2) + 0.2)

    # Full zoom
    fig.subplots_adjust(0,0,1,1)

    return fig

# Scatter plot uncertainties figure
def plot_uncertaintes(X, y, size, fig = None):

    # Create or clear figure
    if fig == None:
        fig = plt.figure(figsize=(5, 4), dpi=100)
    else:
        fig.clear()

    subplot = fig.add_subplot(111)
    X_test = load_Xtest(X, size)

    if uncertainty == "Epistemic":
        certainties, _ = compute_epistemic_uncertainty(X, y, X_test)

        # Just for visual reprensentation 
        # (Remove for real values)
        certainties = np.array(certainties)**1.5

    elif uncertainty == "Aleatoric":
        _, certainties= compute_epistemic_uncertainty(X, y, X_test)

        # Just for visual reprensentation 
        # (Remove for real values)
        certainties = np.array(certainties)

    elif uncertainty == "Discord":
        certainties = compute_discord(X, y, X_test)
    elif uncertainty == "Non-Specificity":
        certainties = compute_nonspe(X, y, X_test)
    else:
        certainties = compute_uncertainty(X, y, X_test)

    if(np.max(certainties) != 0):
        certainties = certainties / np.max(certainties)

    # Map color for certainty
    # colors = np.ones((X_test.shape[0], 3))
    # colors[:,1] = 1 - certainties
    # colors[:,2] = 1 - certainties

    # Remove useless values
    #X_test = X_test[certainties > 0.05]
    #certainties = certainties[certainties > 0.05]
    #colors = colors[certainties > 0.05]

    # Render all points
    #subplot.scatter(X_test[:, 0], X_test[:, 1], color = colors, marker=",", s = 80)
    # indices = np.argsort(certainties)
    # print(X_test.shape, certainties.shape)
    # subplot.contour(X_test[indices, 0:2], certainties[indices])

    gridsize=(int(SIZES[size][0] * 0.68), int(SIZES[size][1] * 0.43))
    subplot.hexbin(X_test[:, 0], X_test[:, 1], C=certainties, gridsize=gridsize, cmap=CM.jet, bins=None)

    # Remove black rectangle
    subplot.spines['top'].set_visible(False)
    subplot.spines['right'].set_visible(False)
    subplot.spines['bottom'].set_visible(False)
    subplot.spines['left'].set_visible(False)

    # Remove ticks form figure
    subplot.set(yticklabels=[], xticklabels=[]) 
    subplot.tick_params(left=False, bottom=False) 

    # Same size for each figure
    subplot.set_xlim(np.min(X[:,0]) + 0.2, np.max(X[:,0]) - 0.2)
    subplot.set_ylim(np.min(X[:,1]) + 0.2, np.max(X[:,1]) - 0.2)

    # Full zoom
    fig.subplots_adjust(0,0,1,1)

    return fig

# Render uncertainty plot
def uncertainty_rendering(fig_uncertainties, size, pic):
    fig_uncertainties = plot_uncertaintes(X, y, size, fig_uncertainties)
    pic.draw()
    return fig_uncertainties

# Render dataset plot
def dataset_rendering(fig_dataset, pic):
    fig_dataset = plot_dataset(X[:,0], X[:,1], y, fig_dataset)
    pic.draw()
    return fig_dataset

# Load data from dataset, scale and format
def load_data(path, v1 = None, v2 = None):

    df = load_dataset("datasets/" + path + ".csv")
    if v1 != None:
        X = df[[v1, v2]]
    else:
        vars = df.columns[(df.columns != 'Unnamed: 0') & (df.columns != 'target')]
        X = df[[vars[0], vars[1]]]
    
    X = preprocessing.scale(X)
    y = df["target"]
    y = preprocessing.LabelEncoder().fit(y).transform(y)

    if reduce_classes:
        y[y != 0] = 1

    features = df.columns.values[(df.columns != 'Unnamed: 0') & (df.columns != 'target')]
    return X, y, features

# Load data from dataset when PCA is required
def load_data_pca(path, method = "PCA"):
    df = load_dataset("datasets/" + path + ".csv")
    vars = df.columns[(df.columns != 'Unnamed: 0') & (df.columns != 'target')]

    X = X = df[vars]
    X = preprocessing.scale(X)
    y = df["target"]
    y = preprocessing.LabelEncoder().fit(y).transform(y)

    if reduce_classes:
        y[y != 0] = 1

    n_components = min(X.shape[0], X.shape[1])
    if method == "PCA":
        pca = PCA(n_components=n_components)
    elif method == "SVD" and n_components != 2:
        X = X + abs(np.min(X))
        pca = TruncatedSVD(n_components=2)
    else:
        X = X + abs(np.min(X))
        pca = TSNE(n_components=2, init='random', learning_rate='auto')
    
    reduced = pca.fit_transform(X)

    X = reduced[:, 0:2]

    return X, y

# List all datasets in "datasets" file
def load_datasets():
    f = []
    for (dirpath, dirnames, filenames) in walk("datasets"):
        for name in filenames:
            f.append(name[:-4])
        break
    return f

# Update feature list
def update_features(window, features):
        window['VAR1_COMBO'].Update(features[0], list(features))
        window['VAR2_COMBO'].Update(features[1], list(features))

# Change visibility for features list
def clear_variables_layout(remove = True):
    if remove:
        window['VAR1_COMBO'].Update(visible = False)
        window['VAR2_COMBO'].Update(visible = False)
        window['VARIABLES'].Update(visible = False)
    else:
        window['VAR1_COMBO'].Update(visible = True)
        window['VAR2_COMBO'].Update(visible = True)
        window['VARIABLES'].Update(visible = True)

# Load the model
classifier = models.load_model("SVM")
# Load the ucertainty
uncertainty = uncertainties.get_uncertaintes()[0]

# Reduce classes to 2 for epistemic uncertainty
reduce_classes= False

# List all datasets
datasets_list = load_datasets()
X, y, features = load_data("Iris")

# Generate app Layout
window = generateLayout(datasets_list, features)

# Dataset Figure
fig_dataset = plot_dataset(X[:,0], X[:,1], y)
pic_dataset = draw_figure(window['DATASET'].TKCanvas, fig_dataset)

# Uncertainties Figure
fig_uncertainties = plot_uncertaintes(X, y, 5)
pic_uncertainties = draw_figure(window['UNCERTAINTIES'].TKCanvas, fig_uncertainties)

continuer = True
while continuer:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    # Uncertainty rendering event
    if event == "RENDER":
        fig_uncertainties = uncertainty_rendering(fig_uncertainties, int(values['RENDER']) - 1, pic_uncertainties)

    # Change dataset
    elif event == "DATASET_COMBO":
        X, y, features = load_data(values['DATASET_COMBO'])
        fig_dataset = dataset_rendering(fig_dataset, pic_dataset)
        fig_uncertainties = uncertainty_rendering(fig_uncertainties, int(values['RENDER']) - 1, pic_uncertainties)
        update_features(window, features)
        window['PCA'].Update(False)
        window['TSNE'].Update(False)
        window['SVD'].Update(False)
        clear_variables_layout(False)

    # Change variables
    elif (event == "VAR1_COMBO") or (event == "VAR2_COMBO"):
        X, y, features = load_data(values['DATASET_COMBO'], values['VAR1_COMBO'], values['VAR2_COMBO'])
        fig_dataset = dataset_rendering(fig_dataset, pic_dataset)
        fig_uncertainties = uncertainty_rendering(fig_uncertainties, int(values['RENDER']) - 1, pic_uncertainties)

    # Change model
    elif (event == "MODELS_COMBO"):
        classifier = models.load_model(values['MODELS_COMBO'])
        fig_uncertainties = uncertainty_rendering(fig_uncertainties, int(values['RENDER']) - 1, pic_uncertainties)

    # Load PCA
    elif (event == "PCA" or event == "TSNE" or event == "SVD"):
        if values["TSNE"] and event == "TSNE":
            X, y = load_data_pca(values['DATASET_COMBO'], "TSNE")
            window['PCA'].Update(False)
            window['SVD'].Update(False)
            clear_variables_layout()
        elif values["PCA"] and event == "PCA":
            X, y = load_data_pca(values['DATASET_COMBO'])
            window['TSNE'].Update(False)
            window['SVD'].Update(False)
            clear_variables_layout()
        elif values["SVD"] and event == "SVD":
            X, y = load_data_pca(values['DATASET_COMBO'], "SVD")
            window['TSNE'].Update(False)
            window['PCA'].Update(False)
            clear_variables_layout()
        else:
            X, y, features = load_data(values['DATASET_COMBO'], values['VAR1_COMBO'], values['VAR2_COMBO'])
            clear_variables_layout(False)
        fig_dataset = dataset_rendering(fig_dataset, pic_dataset)
        fig_uncertainties = uncertainty_rendering(fig_uncertainties, int(values['RENDER']) - 1, pic_uncertainties)

    # Change uncertainty formula
    elif (event == "UNCERTAINTY"):
        uncertainty = values["UNCERTAINTY"]
        fig_uncertainties = uncertainty_rendering(fig_uncertainties, int(values['RENDER']) - 1, pic_uncertainties)

    # Change for total uncertainty
    elif (event == "TOTAL"):
        # Biblio
        window['BIBLIO'].update("")
        # Auto Grid Size
        if values['GRID']:
            window['RENDER'].Update(6)
            values['RENDER'] = 6
        window['MODELS_COMBO'].Update("SVM", disabled = False)
        classifier = models.load_model("SVM")
        window['UNCERTAINTY'].Update(uncertainties.get_uncertaintes()[0], uncertainties.get_uncertaintes())
        uncertainty = uncertainties.get_uncertaintes()[0]
        reduce_classes = False
        if values["PCA"]:
            X, y = load_data_pca(values['DATASET_COMBO'])
        if values["TSNE"]:
            X, y = load_data_pca(values['DATASET_COMBO'], "TSNE")
        if values["SVD"]:
            X, y = load_data_pca(values['DATASET_COMBO'], "SVD")
        else:
            X, y, features = load_data(values['DATASET_COMBO'], values['VAR1_COMBO'], values['VAR2_COMBO'])        
        fig_dataset = dataset_rendering(fig_dataset, pic_dataset)
        fig_uncertainties = uncertainty_rendering(fig_uncertainties, int(values['RENDER']) - 1, pic_uncertainties)

    # Change for epistemic uncertainty
    elif (event == "EPISTEMIC"):
        # Biblio
        window['BIBLIO'].update("Senge R, Bösner S, Dembczynski K, et al (2014) Reliable classification: Learning classifiers that distinguish aleatoric and epistemic uncertainty.  Inf Sci.")
        # Auto Grid Size
        if values['GRID']:
            window['RENDER'].Update(4)
            values['RENDER'] = 4
        window['MODELS_COMBO'].Update("K-NN", disabled = True)
        window['UNCERTAINTY'].Update("Epistemic", ["Epistemic", "Aleatoric"])
        uncertainty = "Epistemic"
        reduce_classes = True
        if values["PCA"]:
            X, y = load_data_pca(values['DATASET_COMBO'], "PCA")
        elif values["TSNE"]:
            X, y = load_data_pca(values['DATASET_COMBO'], "PCA")
        elif values["SVD"]:
            X, y = load_data_pca(values['DATASET_COMBO'], "SVD")
        else:
            X, y, features = load_data(values['DATASET_COMBO'], values['VAR1_COMBO'], values['VAR2_COMBO'])
        fig_dataset = dataset_rendering(fig_dataset, pic_dataset)
        fig_uncertainties = uncertainty_rendering(fig_uncertainties, int(values['RENDER']) - 1, pic_uncertainties)

    # Change for evidential uncertainty
    elif (event == "EVIDENTIAL"):
        # Biblio
        window['BIBLIO'].update("Klir GJ, Wierman MJ (1998) Uncertainty-based information: Elements of generalized information theory. Springer-Verlag")
        # Auto Grid Size
        if values['GRID']:
            window['RENDER'].Update(5)
            values['RENDER'] = 5
        window['MODELS_COMBO'].Update("EK-NN",disabled = True)
        window['UNCERTAINTY'].Update("Discord", ["Discord", "Non-Specificity"])
        uncertainty = "Discord"
        reduce_classes = False
        if values["PCA"]:
            X, y = load_data_pca(values['DATASET_COMBO'])
        elif values["TSNE"]:
            X, y = load_data_pca(values['DATASET_COMBO'], "PCA")
        elif values["SVD"]:
            X, y = load_data_pca(values['DATASET_COMBO'], "SVD")
        else:
            X, y, features = load_data(values['DATASET_COMBO'], values['VAR1_COMBO'], values['VAR2_COMBO'])        
        fig_dataset = dataset_rendering(fig_dataset, pic_dataset)
        fig_uncertainties = uncertainty_rendering(fig_uncertainties, int(values['RENDER']) - 1, pic_uncertainties)

    # Change Grid Size
    elif (event == "GRID"):
        if values['GRID']:
            window['RENDER'].Update(6, visible=False)
        else:
            window['RENDER'].Update(visible=True)

    # Refresh Datasets
    elif (event == "SAVE"):
        fig_dataset.savefig("Dataset")
        fig_uncertainties.savefig("Uncertainties")

    # Refresh Datasets
    elif (event == "REFRESH"):
        datasets_list = load_datasets()
        window['DATASET_COMBO'].Update(values['DATASET_COMBO'], values = datasets_list)
        
    # Contacts page
    elif (event == "CONTACTS"):
        webbrowser.open('https://github.com/ArthurHoa')

# Close app
window.close()