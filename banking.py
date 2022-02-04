import numpy as np
import matplotlib as plt
import pandas as pd

data=pd.read_csv("bank.csv")
x=data.iloc[:,3:13]
y=data.iloc[:,13]

geo=pd.get_dummies(x["Geography"],drop_first=True)
gen=pd.get_dummies(x["Gender"],drop_first=True)

x=pd.concat([x,gen,geo],axis=1)
x=x.drop(["Geography","Gender"],axis=1)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)

#necessary to using sclaling in order to increase the aim of multiplying the weights effectively and quickly
from sklearn.preprocessing import StandardScaler
s=StandardScaler()
xtrain=s.fit_transform(xtrain)
xtest=s.fit_transform(xtest)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#also try using he_uniform since it is better than he_normal
#consider using many layers only for deep NN with dropouts
classifier = Sequential()
classifier.add(Dense(units=10,kernel_initializer='he_normal',activation='relu',input_dim=11))
classifier.add(Dropout(0.3))

classifier.add(Dense(units=20,kernel_initializer='he_normal',activation='relu'))
classifier.add(Dropout(0.4))

classifier.add(Dense(units=15,kernel_initializer='he_normal',activation='relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

mode=classifier.fit(xtrain,ytrain,validation_split=0.33,batch_size=10,nb_epoch=100)

#print(mode.history.keys())

#plt.plot(mode.history['acc'])
#plt.plot(mode.history['val_acc'])
#plt.title('model_accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train','test'],loc='upper left')
#plt.show()



ypred=classifier.predict(xtest)
ypred=(ypred>0.5)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(ytest,ypred)

from sklearn.metrics import accuracy_score
score=accuracy_score(ypred,ytest)