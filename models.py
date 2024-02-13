from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

######################################################
#                                                    #
# Modify this file to append your own models         #
#                                                    #
# Arthur H. 2024                                     #
# https://github.com/ArthurHoa                       #
#                                                    #
######################################################

# List of models
MODELS = ["K-NN", "RandomForest", "SVM", "NaiveBayes"]

# Append your own model
def load_model(model):

    if model == MODELS[0]:
        classifier = KNeighborsClassifier(n_neighbors=7, weights="distance")
    elif model == MODELS[1]:
        classifier = RandomForestClassifier()
    elif model == MODELS[2]:
        classifier = SVC(probability=True)
    elif model == MODELS[3]:
        classifier = GaussianNB()

    return classifier

def get_models():
    return MODELS