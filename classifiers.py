# -*- coding: utf-8 -*-
import modStats
import utils
import cythonMetrics
from time import time

import numpy as np

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def rateClassifiers(df):

    start = time()
    ratings = dict()

    # 50 shows some value improvement compared to 10 for acceptable performance using default KNN from sklearn kit
    utils.Constants().kInKNN = 50
    # .5 gives values a touch better for Y - PCA (X) p stock and does not change much others
    utils.Constants().distDayWeight = 0.5
    #    rateDecisionTree(df, ratings)
    #    print(time() - start, '\n'.join([x + ' : ' + str(ratings[x]) for x in ratings.keys()]))
#    rateDefaultKNN(df, ratings)
#    print(time() - start, '\n'.join([x + ' : ' + str(ratings[x]) for x in ratings.keys()]))
#    rateCustomKNN(df, ratings)
#    print(time() - start, '\n'.join([x + ' : ' + str(ratings[x]) for x in ratings.keys()]))
    rateDefaultLogisticRegression(df, ratings)
    print(time() - start)
    print ('\n'.join([x + utils.SEPARATOR + str(ratings[x][0]) + utils.SEPARATOR + str(ratings[x][1]) for x in ratings.keys()]))

def rateDecisionTree(df, ratings):
    """
    Decision tree to predict discretised movement _ using each stock's variance
    Gives an intuition of Y and features that work the best
    """
    rateOriginalY(df, ratings, tree.DecisionTreeClassifier)
    rateYIncrements(df, ratings, tree.DecisionTreeClassifier)
    rateAllRests(df, ratings, tree.DecisionTreeClassifier)

def rateDefaultKNN(df, ratings):
    """
    Using the standard KNN to predict discretised movement _ using each stock's variance _ 
    K can be set to improve results _ turns up increasing K affects performance significantly with little improvement on the score
    """
    classifierCreator = makeKNeighborsClassifier(utils.Constants().kInKNN)
    rateOriginalY(df, ratings, classifierCreator)
    rateYIncrements(df, ratings, classifierCreator)
    rateAllRests(df, ratings, classifierCreator)
    rateYIncrements(df, ratings, classifierCreator)
    rateAllRests(df, ratings, classifierCreator)

def rateCustomKNN(df, ratings):
    """ 
    KNN algo with customized distance _ weight for days and stocks can be adjusted
    Implementing custom distances comes with two technical limitations: 
        * method is automatically switched to ball tree, which causes severe performance issues for large data sets without regularization
        * the fit and predict functions have to be written in Python and not Cython _ not a major hurdle but good to notice
    """
    utils.Constants().distDayWeight = 0
    utils.Constants().distStockWeight = 0
    utils.Constants().kInKNN = 10
    rateOriginalY(df, ratings, getTunedKNNDistDayStockX)

def rateDefaultLogisticRegression(df, ratings):
    """
    Using the standard LogisticRegression to predict discretised movement _ using each stock's variance _ 
    Penalty is added for regularization purpose
    """
    # l1 and l2 seems to give fairly similar scores l1 only will be used to improve calculation time
    isPenaltyL1 = True
    for c in [0.1]: 
        classifierCreator = makeLogisticRegressionClassifier(isPenaltyL1, c)
        rateOriginalY(df, ratings, classifierCreator)
        rateYIncrements(df, ratings, classifierCreator)
        rateAllRests(df, ratings, classifierCreator)

def rateOriginalY(df, ratings, classifier):
    rateYFromFile(df, ratings, classifier, utils.HEADER_Y_VARIANCE_GROUP, '')

def rateYIncrements(df, ratings, classifier):
    rateYFromFile(df, ratings, classifier, utils.HEADER_Y_VARIANCE_GROUP + utils.LABEL_INCREMENTS, utils.LABEL_INCREMENTS)

def rateYFromFile(df, ratings, classifier, Ylabel, labelSuffix):
    """
    Assess any columns in the csv file in three ways:
        * discretized based on Y's standard deviation at stock level
        * sign
        * absolute discretized change
    """
    # variance groups
    rateGroup(df, ratings, classifier, Ylabel, [utils.HEADER_DAY])
    # sign
    classificationLabel = utils.HEADER_Y_UPDOWN + labelSuffix
    df[classificationLabel] = np.sign(df[Ylabel])
    rateGroup(df, ratings, classifier, classificationLabel, [utils.HEADER_DAY])
    # magnitude
    classificationLabel = utils.HEADER_Y_MAGNITUDE + labelSuffix
    nbVarianceBuckets = utils.Constants().YVarianceBuckets
    df[classificationLabel] = df[Ylabel].apply(lambda x : abs(x // nbVarianceBuckets))
    rateGroup(df, ratings, classifier, classificationLabel, [utils.HEADER_DAY])

def rateAllRests(df, ratings, classifier):
    """
    Assess (Y - predict) for predict coming on linear regression of Xs or F(X)s
    """
    rateRest(df, ratings, classifier, modStats.TYPE_LINEAR_PER_STCK)
    # commented out as it gives worst results than linear per stock and makes less sense:
    # the same function should be applied to the whole feature instead of function fitting per stock
    # if it could make some sense in term of classic regression, it does not for classification features in the general case
#    rateRest(df, ratings, classifier, modStats.TYPE_LSUM_OF_FUNCTIONS_PER_STCK)

def rateRest(df, ratings, classifier, restType):
    """
    Assess (Y - predict) in three ways:
        * discretized based on Y's standard deviation at stock level
        * sign
        * absolute discretized change
    """
    # variance groups
    classificationLabelRest = utils.HEADER_Y_VARIANCE_GROUP + restType
    df[classificationLabelRest] = np.maximum(np.minimum(np.round(np.divide(df[utils.HEADER_Y_REST + restType], df[utils.HEADER_Y_VARIANCE])), 5), -5)
    rateGroup(df, ratings, classifier, classificationLabelRest, [utils.HEADER_DAY])
    # sign
    classificationLabelUpDown = utils.HEADER_Y_UPDOWN + restType
    df[classificationLabelUpDown] = np.sign(df[utils.HEADER_Y_REST + restType])
    rateGroup(df, ratings, classifier, classificationLabelUpDown, [utils.HEADER_DAY])
    # magnitude
    classificationLabelMagnitude = utils.HEADER_Y_MAGNITUDE + restType
    df[classificationLabelMagnitude] = np.multiply(df[classificationLabelRest], df[classificationLabelUpDown])
    rateGroup(df, ratings, classifier, classificationLabelMagnitude, [utils.HEADER_DAY])

def rateGroup(df, ratings, classifier, classificationLabel, additionalParameters):
    """
    for any given target label vector and given classifier, fits the classifier based on 
    """
    ratings[createGroupName(classifier.__name__, classificationLabel, utils.LABEL_X_ORIGINAL, utils.Constants().distDayWeight) ] = rateClassifier(df, additionalParameters + utils.HEADER_X, [(utils.HEADER_DAY, utils.Constants().distDayWeight)] + [(x, 1) for x in utils.HEADER_X], classificationLabel, classifier)
    ratings[createGroupName(classifier.__name__, classificationLabel, utils.LABEL_X_LINEAR, utils.Constants().distDayWeight)] = rateClassifier(df, additionalParameters + utils.getXVectors(df.columns.values, modStats.TYPE_LINEAR), [(utils.HEADER_DAY, utils.Constants().distDayWeight)], classificationLabel, classifier)
    # other sets of features commented as they do not improve the results but are less traightforward to interprete
#    ratings[createGroupName(classifier.__name__, classificationLabel, utils.LABEL_X_ORIGINAL + utils.LABEL_INCREMENTS, utils.Constants().distDayWeight) ] = rateClassifier(df, additionalParameters + [x + utils.LABEL_INCREMENTS for x in utils.HEADER_X], [(utils.HEADER_DAY, utils.Constants().distDayWeight)] + [(x + utils.LABEL_INCREMENTS, 1) for x in utils.HEADER_X], classificationLabel, classifier)
#    ratings[createGroupName(classifier.__name__, classificationLabel, utils.LABEL_X_LSUM_OF_FUNCTIONS, utils.Constants().distDayWeight)] = rateClassifier(df, additionalParameters + utils.getXVectors(df.columns.values, modStats.TYPE_LSUM_OF_FUNCTIONS), [(utils.HEADER_DAY, utils.Constants().distDayWeight)], classificationLabel, classifier)
#    ratings[createGroupName(classifier.__name__, classificationLabel, utils.LABEL_X_LINEAR_PER_STCK, utils.Constants().distDayWeight)] = rateClassifier(df, additionalParameters + utils.getXVectors(df.columns.values, modStats.TYPE_LINEAR_PER_STCK), [(utils.HEADER_DAY, utils.Constants().distDayWeight)], classificationLabel, classifier)
#    ratings[createGroupName(classifier.__name__, classificationLabel, utils.LABEL_X_LSUM_OF_FUNCTIONS_PER_STCK, utils.Constants().distDayWeight)] = rateClassifier(df, additionalParameters + utils.getXVectors(df.columns.values, modStats.TYPE_LSUM_OF_FUNCTIONS_PER_STCK), [(utils.HEADER_DAY, utils.Constants().distDayWeight)], classificationLabel, classifier)

def createGroupName(classifierName, classificationLabel, featuresLabel, kDays):
    return classifierName + utils.SEPARATOR + classificationLabel + utils.SEPARATOR + utils.SEPARATOR + featuresLabel + utils.SEPARATOR + str(kDays)

def rateClassifier(df, Xlabels, XlabelsToNormalise, Ylabel, classifier):
    dfToClassify = df[Xlabels + [Ylabel]].copy(deep = True)
    dfToClassify = dfToClassify.replace([np.inf, -np.inf, 'inf'], 0)
    df_filtered = utils.testNanInDF(dfToClassify, [Xlabels[0], Ylabel])
    dfToClassify = dfToClassify.drop(df_filtered.index.values)

    # normalize vectors which need to be
    if XlabelsToNormalise:
        dfToClassify[[x[0] for x in XlabelsToNormalise]] = StandardScaler(with_mean = True, with_std = True).fit_transform(dfToClassify[[x[0] for x in XlabelsToNormalise]]) * [x[1] for x in XlabelsToNormalise]
    # separate training set and test set then fit the classifier and calculate accuracy
    avgTestScore = 0
    avgTrainScore = 0
    for train, test in utils.getTestAndTrainSample().split(dfToClassify[Xlabels]):
        testClassifier = classifier()
        testClassifier.fit(dfToClassify.loc[train, Xlabels].values, np.array(dfToClassify.loc[train, Ylabel].values, dtype=np.int64))
        avgTrainScore+= testClassifier.score(dfToClassify.loc[train, Xlabels].values, dfToClassify.loc[train, Ylabel].values) / utils.Constants().testSamplingRepeats
        avgTestScore += testClassifier.score(dfToClassify.loc[test, Xlabels].values, dfToClassify.loc[test, Ylabel].values) / utils.Constants().testSamplingRepeats
    return (round(avgTrainScore * 100, 3), round(avgTestScore * 100, 3))

def makeKNeighborsClassifier(kNeighbors):
    def getKNeighborsClassifier():
        return KNeighborsClassifier(n_neighbors = kNeighbors)
    getKNeighborsClassifier.__name__ = 'getKNeighborsClassifier' + str(kNeighbors)
    return getKNeighborsClassifier

def getTunedKNNDistDayStockX():
    return cythonMetrics.TunedKNNDistDayStockX(utils.Constants().kInKNN, utils.Constants().distDayWeight, utils.Constants().distStockWeight)

def makeLogisticRegressionClassifier(isPenaltyL1, C):
    penalty = 'l1' if isPenaltyL1 else'l2'

    def getLogisticRegressionClassifier():
        return LogisticRegression(C=C, penalty=penalty, tol=0.01)
    getLogisticRegressionClassifier.__name__ = 'getLogisticRegressionClassifier' + utils.SEPARATOR + penalty + utils.SEPARATOR + str(C)
    return getLogisticRegressionClassifier
