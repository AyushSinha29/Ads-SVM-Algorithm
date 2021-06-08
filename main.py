import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('Ads.csv')
x = df.iloc[:,2:4].values
y = df.iloc[:,4].values
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
from sklearn.svm import SVC
class_svc = SVC(kernel='linear', random_state=0)
class_svc.fit(x_train, y_train)
y_pred = class_svc.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test,y_pred)
