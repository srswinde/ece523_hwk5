import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.semi_supervised import LabelSpreading

n_features = 2
X, y = make_blobs(n_samples=1000, centers=2, n_features=n_features, random_state=0, cluster_std=1.0)


plt.plot(X[y==1][:,0], X[y==1][:,1], 'r.')
plt.plot(X[y==0][:,0], X[y==0][:,1], 'g.')
plt.show()


LS = LabelSpreading()



