import numpy as np

######################################################
#                                                    #
# Modify this file to append your own methods        #
#                                                    #
# Arthur H. 2024                                     #
# https://github.com/ArthurHoa                       #
#                                                    #
######################################################

# List of uncertainties
UNCERTAINTES = ["Least-Confident", "Entropy", "Gini"]

# Append your own uncertainty
def uncertainty(uncertainty_type, proba):

    uncertainties = []

    if uncertainty_type == UNCERTAINTES[0]:
        uncertainties = shannon_entropy(proba)
    elif uncertainty_type == UNCERTAINTES[1]:
        uncertainties = least_confident(proba)
    elif uncertainty_type == UNCERTAINTES[2]:
        uncertainties = gini(proba)

    return uncertainties

def get_uncertaintes():
    return UNCERTAINTES

# Method 1 - Entropy
def shannon_entropy(proba):
    log_buffer = np.ma.log2(proba)
    find = np.where(np.isnan(log_buffer) != False)
    log_buffer[find[0],find[1]] = 0

    return -1 * np.sum(proba * log_buffer, axis=1)

# Method 2 - LeastConfident
def least_confident(proba):
    return 1 - np.max(proba, axis=1)

# Method 3 - Gini
def gini(proba):
    return 1 - np.sum(proba**2, axis=1)