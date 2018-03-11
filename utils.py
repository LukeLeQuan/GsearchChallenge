# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import ShuffleSplit

HEADER_INDEX = 'Index'; HEADER_MARKET = 'Market'; HEADER_DAY = 'Day'; HEADER_STOCK = 'Stock';
HEADER_X = ['x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']
HEADER_Y = 'y'; HEADER_WEIGHT = 'Weight'
HEADER_Y_VARIANCE = 'yvar'; HEADER_Y_VARIANCE_GROUP = 'yvarGroup'; HEADER_Y_UPDOWN = 'yUpDown'; HEADER_Y_MAGNITUDE = 'yMagnitude'
HEADER_Y_REST = 'yrest'; HEADER_Y_TEST = 'ytest'; HEADER_Y_PREDICT = 'yprdct'


MAIN_LOGGER_NAME = 'mainLog'
TEST_TYPE_K_FOLD = 'KFold'
TEST_TYPE_REPEATED_K_FOLD = 'RepeatedKFold'
TEST_TYPE_RANDOM = 'random'
LABEL_ADD_RATIO = 'addRatio'
LABEL_MUL_RATIO = 'mulRatio'


LABEL_X_ORIGINAL = '[X]'
LABEL_X_LINEAR = '[X PCA]'
LABEL_X_LSUM_OF_FUNCTIONS = '[LF(X) PCA]'
LABEL_X_LINEAR_PER_STCK = '[X PCA p Stck]'
LABEL_X_LSUM_OF_FUNCTIONS_PER_STCK = '[LF(X) PCA p Stck]'

SEPARATOR = "_"

##############################################################################################################################
##############################################################################################################################
###########################################################  utils ###########################################################
####################################### manages constants and train / test strategies  #######################################
##############################################################################################################################

# Singleton/SingletonPattern
defaultConstantSet = {'Yvariance_buckets' : 10, 
              'random_state' : 12883823, 
              'FractionFullSampleForTest' : 0.1, 
              'testSamplingRepeats' : 20, 
              'loggerName' : MAIN_LOGGER_NAME, 
              'testType' : TEST_TYPE_K_FOLD, 
              'nbThreads' : 4, 
              'incrementalFunctionFit' : False, 
              'distDayWeight' : 1, 
              'distStockWeight' : 1000, 
              'kInKNN' : 3}
class Constants:
    class _ConstantSingle:
        def __init__(self, **kwargs):
            self.Yvariance_buckets = self.getParam(kwargs,'Yvariance_buckets')
            self.random_state = self.getParam(kwargs,'random_state')
            self.FractionFullSampleForTest = self.getParam(kwargs,'FractionFullSampleForTest') # set to 10 equivalent to train on 90% of the data and tes on 10%
            self.testSamplingRepeats = self.getParam(kwargs,'testSamplingRepeats') # number of random partitions generated
            self.loggerName = self.getParam(kwargs,'loggerName') # name used as unique identifier for the logger
            self.testType = self.getParam(kwargs,'testType')
            self.nbThreads = self.getParam(kwargs,'nbThreads')
            self.incrementalFunctionFit = self.getParam(kwargs,'incrementalFunctionFit')
            self.distDayWeight = self.getParam(kwargs, 'distDayWeight')
            self.distStockWeight = self.getParam(kwargs, 'distStockWeight')
            self.kInKNN = self.getParam(kwargs, 'kInKNN')
        def __str__(self):
             return repr(self) + self.Yvariance_buckets + self.random_state + self.FractionFullSampleForTest + self.testSamplingRepeats + self.loggerName + self.testType + self.nbThreads + self.incrementalFunctionFit + self.distDayWeight + self.distStockWeight + self.kInKNN
        @staticmethod
        def getParam(paramSet, paramName):
            return paramSet.get(paramName, defaultConstantSet.get(paramName))

    instance = None
    def __init__(self, **kwargs):
        if not Constants.instance:
            Constants.instance = Constants._ConstantSingle(**kwargs)
    @staticmethod
    def tearDown():
        Constants.instance = None

    @property
    def YVarianceBuckets(self,*args,**kwargs):
        return self.instance.Yvariance_buckets
    @YVarianceBuckets.setter
    def YVarianceBuckets(self,*args,**kwargs):
        self.instance.Yvariance_buckets = args[0]
    @property
    def randomState(self,*args,**kwargs):
        return self.instance.random_state
    @property
    def fractionFullSampleForTest(self,*args,**kwargs):
        return self.instance.FractionFullSampleForTest
    @fractionFullSampleForTest.setter
    def fractionFullSampleForTest(self,*args,**kwargs):
        self.instance.FractionFullSampleForTest = args[0]
    @property
    def testSamplingRepeats(self,*args,**kwargs):
        return self.instance.testSamplingRepeats
    @property
    def loggerName(self,*args,**kwargs):
        return self.instance.loggerName
    @property
    def testType(self,*args,**kwargs):
        return self.instance.testType
    @property
    def nbThreads(self,*args,**kwargs):
        return self.instance.nbThreads
    @property
    def incrementalFunctionFit(self,*args,**kwargs):
        return self.instance.incrementalFunctionFit
    @incrementalFunctionFit.setter
    def incrementalFunctionFit(self,*args,**kwargs):
        self.instance.incrementalFunctionFit = args[0]
    @property
    def distDayWeight(self,*args,**kwargs):
        return self.instance.distDayWeight
    @distDayWeight.setter
    def distDayWeight(self,*args,**kwargs):
        self.instance.distDayWeight = args[0]
    @property
    def distStockWeight(self,*args,**kwargs):
        return self.instance.distStockWeight
    @distStockWeight.setter
    def distStockWeight(self,*args,**kwargs):
        self.instance.distStockWeight = args[0]
    @property
    def kInKNN(self,*args,**kwargs):
        return self.instance.kInKNN
    @kInKNN.setter
    def kInKNN(self,*args,**kwargs):
        self.instance.kInKNN = args[0]

def parseFile(fileName):
    df = pd.read_csv(fileName, index_col=0)
    # remove rows with a nan parameter _ nan == * systematically returns false hence the filtering using (x1 != x1) | ...
    df_filtered = testNanInDF(df, HEADER_X + [HEADER_Y, HEADER_WEIGHT])
    df = df.drop(df_filtered.index.values)
    df = df.reset_index(drop=True) #to avoid missing rows in the index that cause K-fold to crash
    return df

def getXVectors(columns, regressionType):
    l = len(regressionType)
    return [x for x in columns if (x[-l:] == regressionType) and (str.lower(x[0]) == 'x')]

def getTestAndTrainSample():
    random_state = Constants().randomState
    fraction = Constants().fractionFullSampleForTest
    repeats = Constants().testSamplingRepeats
    testType = Constants().testType

    if testType == TEST_TYPE_K_FOLD:
        return KFold(n_splits=int(1 / fraction), random_state=random_state)
    if testType == TEST_TYPE_REPEATED_K_FOLD:
        return RepeatedKFold(n_splits=int(1 / fraction), n_repeats=repeats, random_state=random_state)
    if testType == TEST_TYPE_RANDOM:
        return ShuffleSplit(n_splits=repeats, test_size=.25, random_state=random_state)

def getShapesToPlot(i : int):
    # 'o' removed to keep it for special display
    shapes = ['v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']
    return shapes[i % len(shapes)]

def testNanInDF(df, headers):
    return df.replace([np.inf, -np.inf], np.nan).query(''.join(['(' + x + ' != ' + x + ') | ' for x in headers ])[:-2])

def addYVarianceGroups(df, Ylabel, groupHeaders, YvarLabel, YvarGroupLabel):
    grouped = df.groupby(groupHeaders)
    df[YvarLabel] = 0
    df[YvarGroupLabel] = 0
    for name, subdf in grouped:
        subsetVariance = subdf[Ylabel].std()
        # for each value of Y, expresses it relatively to the variance
        # the value will be allocated to one of (4 k + 2) buckets where k is set using 'get_Yvariance_buckets'
        # range [0; sigma], [sigma; 2 sigma], [2 sigma; 3 sigma] and their opposite are divided in k, the rest goes in two other buckets
        df.loc[subdf[Ylabel].index,YvarGroupLabel] = subdf[Ylabel].apply(lambda x : np.sign(x) * min(abs(x) // (subsetVariance / Constants().YVarianceBuckets), 3 * Constants().YVarianceBuckets + 1))
        df.loc[subdf[Ylabel].index,YvarLabel] = subsetVariance
