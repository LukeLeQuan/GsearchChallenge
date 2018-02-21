# -*- coding: utf-8 -*-
import modStats
import utils

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def rateClassifiers(df):
    ratings = dict()
    rateVarianceGroup(df, ratings, tree.DecisionTreeClassifier)
    print(ratings)

def rateVarianceGroup(df, ratings, classifier):
    classificationLabel = utils.HEADER_Y_VARIANCE_GROUP
    ratings[classifier.__name__ + classificationLabel + utils.LABEL_X_ORIGINAL ] = rateClassifier(df, utils.HEADER_X, [utils.HEADER_X[10]], classificationLabel, classifier)
    ratings[classifier.__name__ + classificationLabel + utils.LABEL_X_LINEAR] = rateClassifier(df, utils.getXVectors(df.columns.values, modStats.TYPE_LINEAR), [], classificationLabel, classifier)
    ratings[classifier.__name__ + classificationLabel + utils.LABEL_X_LSUM_OF_FUNCTIONS] = rateClassifier(df, utils.getXVectors(df.columns.values, modStats.TYPE_LSUM_OF_FUNCTIONS), [], classificationLabel, classifier)
    ratings[classifier.__name__ + classificationLabel + utils.LABEL_X_LINEAR_PER_STCK] = rateClassifier(df, utils.getXVectors(df.columns.values, modStats.TYPE_LINEAR_PER_STCK), [], classificationLabel, classifier)
    ratings[classifier.__name__ + classificationLabel + utils.LABEL_X_LSUM_OF_FUNCTIONS_PER_STCK] = rateClassifier(df, utils.getXVectors(df.columns.values, modStats.TYPE_LSUM_OF_FUNCTIONS_PER_STCK), [], classificationLabel, classifier)

def rateClassifier(df, Xlabels, XlabelsToNormalise, Ylabel, classifier):
    dfToClassify = df[Xlabels + [Ylabel]].copy(deep = True)
    df_filtered = utils.testNanInDF(dfToClassify, Xlabels + [Ylabel])
    dfToClassify = dfToClassify.drop(df_filtered.index.values)

    # normalize vectors which need to be
    if XlabelsToNormalise:
        dfToClassify[XlabelsToNormalise] = StandardScaler(with_mean = True, with_std = True).fit_transform(dfToClassify[XlabelsToNormalise])
    # separate training set and test set then fit the classifier and calculate accuracy
    avgAccuracy = 0
    for train, test in utils.getTestAndTrainSample().split(df[Xlabels]):
        testClassifier = classifier()
        testClassifier.fit(df.loc[train, Xlabels].values, df.loc[train, Ylabel].values)
        predictions = testClassifier.predict(df.loc[test, Xlabels].values)
        avgAccuracy += accuracy_score(df.loc[test, Ylabel].values, predictions) / utils.Constants().testSamplingRepeats
    return avgAccuracy


