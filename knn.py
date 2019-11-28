import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

number_of_neighbors = 7
y_pred = np.zeros(len(y_test), dtype = int)

for i in range(len(X_test)):
  dist = np.zeros([len(X_train),2] , dtype=float)
  for j in range(len(X_train)):
    dis=0
    for w in range(len(X_train[0])):
      dis = dis+(X_train[j][w] - X_test[i][w])**2
    dis = math.sqrt(dis)
    dist[j]=(dis,y_train[j])
  dist = sorted(dist, key=lambda tup: tup[0])
  freq = np.zeros(number_of_neighbors, dtype=int)
  for j in range(number_of_neighbors):
    freq[int(dist[j][1])] = freq[int(dist[j][1])] + 1
  ind,pr = 0,0
  for j in range(len(freq)):
    if(freq[j]>pr):
      ind=j
      pr=freq[j]  
  y_pred[i] = ind

accuracy = accuracy_score(y_test, y_pred)
print('%.2f'%(accuracy*100),"%")
