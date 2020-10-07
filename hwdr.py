#importing the requried lib for the hand digit recognition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#loading the train and test csv data file from the local disk 
train_data = pd.read_csv(r"C:\Users\NEW PC\Desktop\hand digit\train.csv",encoding='utf-8')
train_data.tail()
X = train_data.drop(['label'],axis=1)
Y = train_data['label'] 
X_train, Y_train = X[0:21000], Y[0:21000]
X_Test, Y_Test = X[21000:], Y[21000:]
grid_data = X_train.values[40].reshape(28,28)
plt.imshow(grid_data,interpolation=None,cmap="gray")
plt.title(Y_train.values[40])
plt.show()
#buliding a training model
model = RandomForestClassifier()
model.fit(X_train,Y_train)
prediction = model.predict(X_Test)
print("model score/accuracy is" + str(accuracy_score(Y_Test,prediction)))
print("confusion matrix \n" + str(confusion_matrix(Y_Test,prediction)))
prediction_test = model.predict(X_Test)
index = 1
print("predicted" + str(prediction_test[index]))
plt.imshow(X_Test.iloc[index].values.reshape((28,28)),cmap='gray')
