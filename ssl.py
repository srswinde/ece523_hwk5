import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random


class main:
    def __init__( self, percent_labeled=0.1 ):
        self.pu = 1 - percent_labeled

    def __call__( self ):
        n_features = 2
        x_true, y_true = make_blobs(
            n_samples=1000,
            centers=2,
            n_features=n_features,
            random_state=0,
            cluster_std=1.1)

        # make experimental partially labeled data
        x_exp, y_exp = x_true[:], y_true[:]

        # randomly take away 80% of the labels
        unlabeled = random.sample(range(y_exp.size), int(self.pu*y_exp.size))
        y_exp[ unlabeled] = -1

        fig = plt.figure()
        self.plot(
            x_exp, y_exp,
            fig.add_subplot(
                5, 2, 1,
                title="Run #1", xlabel="feature1", ylabel="feature2") )
        pgood = 0
        counter = 2
        for counter in range(2, 9):

            pgood = y_exp[y_exp != -1].size/y_exp.size
            print( pgood )
            cls = KNeighborsClassifier( n_neighbors=10 )
            cls.fit( x_exp[y_exp != -1], y_exp[y_exp != -1] )
            prd = np.zeros(x_exp.shape, dtype=int)
            prd[:] = -2  # -2 has already been classified

            # if all datapoints have been classified
            if not np.any(y_exp == -1):
                break
            prd[y_exp == -1] = cls.predict_proba( x_exp[y_exp == -1] )


            cls0 = np.where( prd[:, 0] > 0.95 )
            cls1 = np.where( prd[:, 1] > 0.95 )
            y_exp[cls0] = 0
            y_exp[cls1] = 1

            try:
                self.plot(
                    x_exp, y_exp,
                    fig.add_subplot(
                        5, 2, counter,
                        title="Run #{}".format(counter),
                        xlabel="feature1", ylabel="feature2"),
                        )
            except ValueError as err:
                print(err)
                break

            counter += 1

        self.x_true, self.y_true = x_true, y_true
        self.x_exp, self.y_exp = x_exp, y_exp
        self.prd = prd
        self.cls = cls



    def plot(self, X, y, sp=None, **kwargs):

        if sp is None:
            pltr = plt
        else:
            pltr = sp
        pltr.plot( X[y == 1][:, 0], X[y == 1][:, 1], 'r.',  label="Class 0")
        pltr.plot( X[y == 0][:, 0], X[y == 0][:, 1], 'g.',  label="Class 1")
        pltr.plot(
            X[y == -1][:, 0], X[y == -1][:, 1],
            'k.',
            label="Unknown Class")



doit = main(0.10)
doit()
plt.tight_layout()
plt.legend()
plt.show()

doit = main(0.25)
doit()
plt.tight_layout()
plt.legend()
plt.show()
