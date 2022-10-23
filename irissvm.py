#using svm and RBF
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris') # returns a pandas dataframe

X_iris = iris.drop('species', axis=1)  
y_iris = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state = 0)

clf = SVC(kernel='rbf', C=1).fit(X_train, y_train)
print('iris dataset')
print('Accuracy of RBF SVC classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of RBF SVC classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

import seaborn as sns
iris = sns.load_dataset('iris') # returns a pandas dataframe
import pandas as pd
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_iris = iris.drop('species', axis=1)  
y_iris = iris['species']

xtrain, xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state = 0)
clf = SVC(kernel='rbf', C=1).fit(xtrain, ytrain)
print('Iris dataset')
print('Accuracy of RBF SVC classifier on training set: {:.2f}'
     .format)

from sklearn.metrics import accuracy_score
a = accuracy_score(y_test, y_model)
st.write(a)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_model))

# Confusion Matrix
from sklearn.metrics import confusion_matrix 
confusion_matrix(y_test, y_model)

#Confusion Matrix
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
cf=confusion_matrix = metrics.confusion_matrix(y_test, y_model)
st.write(cf)


print(confusion_matrix)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=np.unique(y_iris))
fig=plt.figure(figsize=(10,4))
cm_display.plot()
plt.show()
st.pyplot(fig)

from sklearn.metrics import classification_report
# F1 score = 2 / [ (1/precision) + (1/ recall)]
print(classification_report(y_test, y_model))
