import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('bank.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

geography=pd.get_dummies(X["Geography"],)
gender=pd.get_dummies(X['Gender'],)


X=pd.concat([X,geography,gender],axis=1)

X=X.drop(['Geography','Gender'],axis=1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid



def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=xtrain.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=0)


layers = [(20,), (40, 20), (45, 30, 15)]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)

grid_result = grid.fit(xtrain,ytrain)
print(grid_result.best_score_,grid_result.best_params_)

ypred=grid.predict(xtest)
pred_y=(ypred>0.5)
from sklearn.metrics import confusion_matrix,accuracy_score


cm=confusion_matrix(pred_y,ytest,)
score=accuracy_score(pred_y,ytest)