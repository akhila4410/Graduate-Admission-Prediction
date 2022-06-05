import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

#read the csv file
data = pd.read_csv("Admission_Predict.csv")

#drop the serial column
data.drop(["Serial No."], axis=1, inplace=True)
x = data.iloc[:, 1].values
y = data.iloc[:, 7].values

#split into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 4, random_state=0)
y_train1 = [1 if each > 0.8 else 0 for each in y_train]
y_test1 = [1 if each > 0.8 else 0 for each in y_test]

#convert ytrain1 and ytest1 into mnumpy arrays
y_train1 = np.array(y_train1)
y_test1 = np.array(y_test1)

#Scale the data
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

#Logistic regression
log_reg = LogisticRegression(random_state=0)
log_reg = log_reg.fit(x_train, y_train1)
ypred = log_reg.predict(x_test)


cm = confusion_matrix(ypred, y_test1)
acc = accuracy_score(ypred, y_test1)


x_set, y_set = x_test, y_test1
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))

plt.contourf(x1, x2, log_reg.predict(np.array([x1.flatten(), x2.flatten()]).T).reshape(x1.shape), alpha=0.95,
             cmap=ListedColormap(('yellow', 'blue')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('black', 'red'))(i), label=j)

plt.xlabel('scores')
plt.ylabel('chance of admit')
plt.title('Logistic Regression')
plt.legend()
