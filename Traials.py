import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

with open('X_train.pickle', 'rb') as X_training, open('y_train.pickle', 'rb') as y_training, open('X_test.pickle','rb') as X_test, open('y_test.pickle', 'rb') as y_test:
    Xtrain = pickle.load(X_training)
    ytrain = pickle.load(y_training)
    Xtest = pickle.load(X_test)
    ytest = pickle.load(y_test)

results=[]
model=LinearRegression()
model.fit(Xtrain, ytrain)
r2 = r2_score(ytest, model.predict(Xtest))
results.append(r2)
name=['Linear Regression']

score = pd.DataFrame(results, index=name)
score.columns = ['R2']
score.sort_values(by='R2', ascending=False)

print(score)