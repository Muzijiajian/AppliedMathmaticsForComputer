#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Create 40 separable points
# Translates slice objects to concatenation along the first axis, np.r_ & np.c_
X_train = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
X_label = [0] * 20 + [1] * 20
# print X_train
# print X_label

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X_train, X_label)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5,5)   #defalut will get 50 ndarray value
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors

b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])
# print clf.support_vectors_
# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none')
plt.scatter(X_train[:, 0], X_train[:, 1], c=X_label, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()