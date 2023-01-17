import pickle
from Model_dev import *

""""##################### Importing Data  #####################################"""

pickle_in = open('X_train.pickle', 'rb')
X_training = pickle.load(pickle_in)

pickle_in = open('y_train.pickle', 'rb')
y_training = pickle.load(pickle_in)

pickle_in = open('X_test.pickle', 'rb')
X_test = pickle.load(pickle_in)

pickle_in = open('y_test.pickle', 'rb')
y_test = pickle.load(pickle_in)

'''######################## Model Check #####################################'''

reg = RegressionModels(X_training, y_training, X_test, y_test)

table=reg.fit_models()

'''############################# Printing Result ###########################'''

print(table.sort_values(by='R2', ascending=False))

























