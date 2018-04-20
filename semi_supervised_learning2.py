import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random
import copy

class TrainData:
    """A convenient way to grab data without saving them to disk."""
    urls = (
        "https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/thyroid_train.csv",
        "https://github.com/gditzler/UA-ECE-523-Sp2018/raw/master/data/echocardiogram_train.csv",
        "https://github.com/gditzler/UA-ECE-523-Sp2018/raw/master/data/breast-cancer_train.csv",
        "https://github.com/gditzler/UA-ECE-523-Sp2018/raw/master/data/congressional-voting_train.csv",
        "https://github.com/gditzler/UA-ECE-523-Sp2018/raw/master/data/conn-bench-sonar-mines-rocks_train.csv",
        "https://github.com/gditzler/UA-ECE-523-Sp2018/raw/master/data/cylinder-bands_train.csv",
        "https://github.com/gditzler/UA-ECE-523-Sp2018/raw/master/data/echocardiogram_train.csv",
        "https://github.com/gditzler/UA-ECE-523-Sp2018/raw/master/data/haberman-survival_train.csv",
        "https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/hayes-roth_train.csv",
        "https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/heart-hungarian_train.csv",
        "https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/hill-valley_train.csv",
        "https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/horse-colic_train.csv",
        "https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/ionosphere_train.csv",
        "https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/image-segmentation_train.csv",
        "https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/mammographic_train.csv",
        "https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/monks-1_train.csv",
    )
    cache =  None
    def __getitem__( self, key ):
        if self.cache is None:
            self.cache = {}

        if key in self.cache:
            data = self.cache[key][:]

        else:
            data = pd.read_csv( self.urls[key], header=None ).as_matrix()
            self.cache[key] = data[:]


        return copy.deepcopy( data )

    def __iter__(self):
        for ii in range( len( self.urls ) ):
            yield copy.deepcopy( self.__getitem__(ii) )


    def __call__( self ):
        """Cache the data"""
        for data in self.__iter__():
            pass

    def xy( self, key ):
        data = self.__getitem__( key )
        x, y = data[:, :-1][:], data[:, -1][:]
        return copy.deepcopy( x ), copy.deepcopy( y )

    def __len__(self):
        return len(self.urls)


class TestData( TrainData ):
    "In case we want the test data"
    def __init__( self ):
        self.urls = [ url.replace("_train", "_test") for url in self.urls ]



class main:

    def __init__( self, percent_labeled=0.1 ):
        self.pu = 1 - percent_labeled


    def getdata( self, key ):
        self.t = TrainData()
        return self.t.xy(key)

    def __call__( self, data=0 ):
        n_features = 2
        # x_true, y_true = make_blobs(
            # n_samples=1000,
            # centers=2,
            # n_features=n_features,
            # random_state=0,
            # cluster_std=1.1)


        x_true, y_true = self.getdata( data )

        # make experimental partially labeled data
        x_exp, y_exp = x_true[:], y_true[:]

        # randomly take away self.pu % of the labels
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
        n_classes = int( np.max(y_exp+1) )
        for counter in range(2, 9):

            pgood = y_exp[y_exp != -1].size/y_exp.size
            print( pgood )
            clsfer = KNeighborsClassifier( n_neighbors=2 )
            clsfer.fit( x_exp[y_exp != -1], y_exp[y_exp != -1] )
            prd = np.zeros((x_exp.shape[0], n_classes ), dtype=int)
            prd[:] = -2  # -2 has already been classified

            # if all datapoints have been classified
            if not np.any(y_exp == -1):
                break
            prd[y_exp == -1] = clsfer.predict_proba( x_exp[y_exp == -1] )


            for cls_i in range(n_classes):
                classified = np.where( prd[:, cls_i] > 0.99 )
                y_exp[classified] = cls_i

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



    def plot(self, X, y, sp=None, **kwargs):
        return
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


