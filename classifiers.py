# -*- coding: utf-8 -*-
import modStats
import utils
import cythonMetrics
from time import time

import numpy as np

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def rateClassifiers(df):

    start = time()

    ratings = dict()
    # decision tree classifier
#    rateVarianceGroup(df, ratings, tree.DecisionTreeClassifier)
#    rateUpDownGroup(df, ratings, tree.DecisionTreeClassifier)
#    rateMoveMagnitudeGroup(df, ratings, tree.DecisionTreeClassifier)
    # default KNN
#    rateVarianceGroup(df, ratings, KNeighborsClassifier)
#    rateUpDownGroup(df, ratings, KNeighborsClassifier)
#    rateMoveMagnitudeGroup(df, ratings, KNeighborsClassifier)
    # KNN algo with home made distance _ weight for days and stocks can be adjusted
    utils.Constants().distDayWeight = 0
    utils.Constants().distStockWeight = 0
    utils.Constants().kInKNN = 63
    rateVarianceGroup(df, ratings, getTunedKNNDistDayStockX)
    print(time() - start, ratings)
    utils.Constants().kInKNN = 3
    rateUpDownGroup(df, ratings, getTunedKNNDistDayStockX)
    print(time() - start, ratings)
    utils.Constants().kInKNN = 4
    rateMoveMagnitudeGroup(df, ratings, getTunedKNNDistDayStockX)
    print(time() - start, ratings)

# assess Y depending on X
def rateVarianceGroup(df, ratings, classifier):
    rateGroup(df, ratings, classifier, utils.HEADER_Y_VARIANCE_GROUP, [utils.HEADER_DAY, utils.HEADER_STOCK])

# assess sign(Y) depending on X
def rateUpDownGroup(df, ratings, classifier):
    classificationLabel = utils.HEADER_Y_UPDOWN
    df[classificationLabel] = np.sign(df[utils.HEADER_Y_VARIANCE_GROUP])
    rateGroup(df, ratings, classifier, classificationLabel, [utils.HEADER_DAY, utils.HEADER_STOCK])

# assess sign(Y) depending on X
def rateMoveMagnitudeGroup(df, ratings, classifier):
    classificationLabel = utils.HEADER_Y_MAGNITUDE
    nbVarianceBuckets = utils.Constants().YVarianceBuckets
    df[classificationLabel] = df[utils.HEADER_Y_VARIANCE_GROUP].apply(lambda x : abs(x // nbVarianceBuckets))
    rateGroup(df, ratings, classifier, classificationLabel, [utils.HEADER_DAY, utils.HEADER_STOCK])

def rateGroup(df, ratings, classifier, classificationLabel, additionalParameters):
    ratings[classifier.__name__ + classificationLabel + utils.LABEL_X_ORIGINAL ] = rateClassifier(df, additionalParameters + utils.HEADER_X, [utils.HEADER_X[10]], classificationLabel, classifier)
#    ratings[classifier.__name__ + classificationLabel + utils.LABEL_X_LINEAR] = rateClassifier(df, additionalParameters + utils.getXVectors(df.columns.values, modStats.TYPE_LINEAR), [], classificationLabel, classifier)
#    ratings[classifier.__name__ + classificationLabel + utils.LABEL_X_LSUM_OF_FUNCTIONS] = rateClassifier(df, additionalParameters + utils.getXVectors(df.columns.values, modStats.TYPE_LSUM_OF_FUNCTIONS), [], classificationLabel, classifier)
#    ratings[classifier.__name__ + classificationLabel + utils.LABEL_X_LINEAR_PER_STCK] = rateClassifier(df, additionalParameters + utils.getXVectors(df.columns.values, modStats.TYPE_LINEAR_PER_STCK), [], classificationLabel, classifier)
#    ratings[classifier.__name__ + classificationLabel + utils.LABEL_X_LSUM_OF_FUNCTIONS_PER_STCK] = rateClassifier(df, additionalParameters + utils.getXVectors(df.columns.values, modStats.TYPE_LSUM_OF_FUNCTIONS_PER_STCK), [], classificationLabel, classifier)

def rateClassifier(df, Xlabels, XlabelsToNormalise, Ylabel, classifier):
    dfToClassify = df[Xlabels + [Ylabel]].copy(deep = True)
    dfToClassify = dfToClassify.replace([np.inf, -np.inf, 'inf'], 0)
    df_filtered = utils.testNanInDF(dfToClassify, [Xlabels[0], Ylabel])
    dfToClassify = dfToClassify.drop(df_filtered.index.values)

    # normalize vectors which need to be
    if XlabelsToNormalise:
        dfToClassify[XlabelsToNormalise] = StandardScaler(with_mean = True, with_std = True).fit_transform(dfToClassify[XlabelsToNormalise])
    # separate training set and test set then fit the classifier and calculate accuracy
    avgAccuracy = 0
    for train, test in utils.getTestAndTrainSample().split(dfToClassify[Xlabels]):
        testClassifier = classifier()
        testClassifier.fit(dfToClassify.loc[train, Xlabels].values, dfToClassify.loc[train, Ylabel].values)
        predictions = testClassifier.predict(dfToClassify.loc[test, Xlabels].values)
        avgAccuracy += accuracy_score(dfToClassify.loc[test, Ylabel].values, predictions) / utils.Constants().testSamplingRepeats
    return round(avgAccuracy * 100, 3)

def getTunedKNNDistDayStockX():
    return cythonMetrics.TunedKNNDistDayStockX(utils.Constants().kInKNN, utils.Constants().distDayWeight, utils.Constants().distStockWeight)