# -*- coding: utf-8 -*-
"""
Created: Jul 2020

@author: Angelika
"""
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing data (without changes inside data)
data = pd.read_csv(r'C:\Users\angel\AppData\Local\Programs\Python\Sieci VSCode\Kaggle\data.csv') 
del data['Unnamed: 32']

X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values #'A', 'A', 'B'...

print("y before:", y)
#encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y) #'1', '1', '0'...
print("y after:", y)

#spliting the dataset into the Training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#now we have prepared data, we will import keras and its packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

#initialising the ANN
classifier = Sequential()

#adding the input layer and the first hidden layer
classifier.add(Dense(16, activation = 'relu', input_dim=30)) # init = 'uniform' 
'''
input_dim - number of columns of the datasets
output_dim (units or default units) - number of outputs to be fed to the next layer, if any
activation - activation function which is ReLU in this case
init - the way in which weights should be provided to an ANN
'''

#adding dropout to prevent overfitting
classifier.add(Dropout(0.1))

#adding the output layer
classifier.add(Dense(1, activation = 'sigmoid'))
''' output_dim (units) is 1 as we want only 1 output from the final layer'''

#compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN to the Training set
classifier.fit(X_train, y_train, epochs = 150, batch_size = 100) #set to epochs = 150!!

#predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))

sns.heatmap(cm,annot=True)
plt.savefig('h.png')










