# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


def KNNgroups(df, Xlabels, XlabelsToNormalise, Ylabels):
    dfToClassify = df.copy(deep = True)
    # normalize vectors which need to be
    for x in XlabelsToNormalise:
        dfToClassify[x] = preprocessing.normalize(df[x])
    # run KNN
    
