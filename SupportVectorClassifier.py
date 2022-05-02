import numpy as np
from numpy.linalg import norm
from scipy import rand 
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpimg
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.feature_selection import RFE
from skimage.transform import resize  
from skimage import filters, io
from skimage import exposure
from skimage import color
from skimage.feature import hog  
from PIL import Image
from sklearn.datasets import load_digits 
from sklearn.svm import SVC
import os
import pickle 
""" ======================  Function definitions ========================== """
def charToSymbol(firstChar):
    switcher = {
        'a': 10, #addition
        's': 11, #subtraction
        'm': 12, #multiplication
        'd': 13, #division
        'r': 15  #random image
    }
    return switcher.get(firstChar, -1) #not a valid filename

def getImageType(file):
    firstChar = file[0]
    if not firstChar.isdigit():  # char isn't a digit (0-9)
        firstChar = charToSymbol(firstChar) #get value for corresponding math symbol
    return firstChar

#set up Vectors 

if __name__ == '__main__':
    
    x1_vector = []
    y_vector = []
    count = 0
    ext = ('.jpg')
    for files in os.listdir('D:\code\project-team-leafs'):
        if files.endswith(ext):
            #load data
                im = mpimg.imread(files) 
                label = getImageType(files) # label the image
                y_vector.append(label) # add label to the y vector
                #grayscale, normalize, and feature_extractor (Histogram of oriented Gradients) 
                resized_im = color.rgb2gray(resize(im,(300,300)))
                bw_im = (resized_im < (filters.threshold_yen(resized_im)))
                bw_im = bw_im.astype(np.float32) 
                fd = hog(bw_im, orientations = 9, pixels_per_cell=(16,16),cells_per_block=(8,8),block_norm='L2')
                x1_vector = np.append(x1_vector,fd, axis=0)
                count = count+1
        else:
                continue
filename = 'final_model.sav'
load = True
#    n_samples = len(digits.images)
"""
    #reshape 
    x = []
    hog_vect = []
    for i in range(1797) :
        gscale = color.rgb2gray(digits.images[i])
        resized_im = resize(gscale,(300,300))
        fd = hog(resized_im, orientations = 9, pixels_per_cell=(4,4),cells_per_block=(2,2),block_norm='L1')
        hog_vect = np.append(hog_vect, fd, axis=0)
    x2 = hog_vect.reshape(n_samples,-1)
    y2 = digits.target 
x = x1_vector.reshape(len(y_vector),-1) 
y = y_vector 
"""
x = x1_vector.reshape(len(y_vector),-1) 
x_train, x_test, y_train, y_test = train_test_split(x, y_vector ,test_size=.10, random_state = 1)
#a_train, a_test, b_train, b_test = train_test_split(x, y ,test_size=.90, random_state = 36)    

    #test using Cross validation 
        #using training data only, perform cross validation to determine best hyperparameters (2x - once for each normalization)
"""
    C = [0.1, 0.5, 1, 5, 10, 15, 20]
    gamma  = ['scale', 'auto']
    degree = [2, 4, 5, 6, 8, 10]
    kernel = ['poly', 'rbf', 'linear']
    random_grid = {
                'C' : C,
                'gamma': gamma,
                'degree': degree,
                'kernel': kernel
              }
    cv = KFold(n_splits =10, random_state=1, shuffle = True)
    rf_random = GridSearchCV(SVC(), param_grid=random_grid, cv = cv)
    rf_random.fit(x_train, y_train)
    print(rf_random.best_params_)
    """
#run tuned classifier
if(load):
    loaded_model = pickle.load(open(filename, 'rb'))
    print(loaded_model.score(x_test, y_test))
else :
    digits = load_digits()
    cv = KFold(n_splits =10, random_state=1, shuffle = True)
    svc = SVC(kernel="linear", C=1)
    svc.fit(x_train, y_train)
    print(svc.score(x_test, y_test))
    pickle.dump(svc, open(filename, 'wb'))
