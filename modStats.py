# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import logging

import utils

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

TYPE_LINEAR = 'linear'
TYPE_LSUM_OF_FUNCTIONS = 'lsumOfFuncs'
TYPE_LINEAR_PER_MKT = 'linearPerMarket'
TYPE_LSUM_OF_FUNCTIONS_PER_MKT = 'lsumOfFuncsPerMarket'
TYPE_LINEAR_PER_STCK = 'linearPerStock'
TYPE_LSUM_OF_FUNCTIONS_PER_STCK = 'lsumOfFuncsPerStock'

class ArrayFunctions():
    # shifts and logs to avoid definitio problems
    @staticmethod
    def identity(x):
        return x
    @staticmethod
    def exp(x):
        return np.exp(x / np.max(x))
    @staticmethod
    def log(x):
        return np.log(x + 0.0000000000001)
    @staticmethod
    def digit80Percent(x):
        return np.greater(x, np.percentile(x, 80)).astype(int)
    @staticmethod
    def digit90Percent(x):
        return np.greater(x, np.percentile(x, 90)).astype(int)
    @staticmethod
    def addRatio(x):
        result = x - np.roll(x, 1)
        return result
    @staticmethod
    def multRatio(x):
        result = x / (np.roll(x, 1) + 0.0000000000001)
        return result
    @staticmethod
    def expOfMultRatio(x):
        return ArrayFunctions.exp(ArrayFunctions.multRatio(x))
    @staticmethod
    def logOfMultRatio(x):
        return ArrayFunctions.log(ArrayFunctions.multRatio(x))
    @staticmethod
    def digit80PercentOfAddRatio(x):
        return ArrayFunctions.digit80Percent(ArrayFunctions.addRatio(x))
    @staticmethod
    def digit90PercentOfAddRatio(x):
        return ArrayFunctions.digit90Percent(ArrayFunctions.addRatio(x))
    @staticmethod
    def digit80PercentOfMultRatio(x):
        return ArrayFunctions.digit80Percent(ArrayFunctions.multRatio(x))
    @staticmethod
    def digit90PercentOfMultRatio(x):
        return ArrayFunctions.digit90Percent(ArrayFunctions.multRatio(x))
    @staticmethod
    def isRatio(func):
        return func.__name__[-5:] == 'Ratio'
    @staticmethod
    def assignVarianceBasedGroup(x, xVar, nbVarianceBuckets, nbBucketsPerVarianceUnit):
        nbFullBuckets = nbVarianceBuckets // (nbBucketsPerVarianceUnit * 2)
        maxNbVar= (nbVarianceBuckets // 2 ) - nbFullBuckets * nbBucketsPerVarianceUnit + nbFullBuckets - 1
        bins = [x for x in range(- maxNbVar, - nbFullBuckets)]
        bins += [x / nbBucketsPerVarianceUnit for x in range(- nbFullBuckets * nbBucketsPerVarianceUnit, nbFullBuckets * nbBucketsPerVarianceUnit)]
        bins += [x for x in range(nbFullBuckets, maxNbVar + 1)]

        return np.digitize(np.divide(x, xVar / nbBucketsPerVarianceUnit), bins) - (nbVarianceBuckets // 2)

# virtual class packaging different predictors
class mktPredictor:
    def estimate(self, dfToPredict, Xlabels, Ylabel):
        raise NotImplementedError()
    @staticmethod
    def predictorFactory(type, df, Xlabels, Ylabel, Wlabel, MktLabel, StckLabel, displayCharts = False):
        ##############################################################################################################################
        ##############################################################################################################################
        ############################  factory methods to return the most accurate predictor for each type ############################
        ##############################################################################################################################
        ##############################################################################################################################
        if type == TYPE_LINEAR:
            return LinearPredictor.getAccurateLinearPredictor(df, Xlabels, Ylabel, Wlabel, displayCharts)
        if type == TYPE_LSUM_OF_FUNCTIONS:
            return LsumOfFunctionsPredictor.getBestLsumOfFunctionsPredictor(df, Xlabels, Ylabel, Wlabel, displayCharts)
        if type == TYPE_LINEAR_PER_MKT:
            return LinearPredictorPerSegment(df, Xlabels, MktLabel, Ylabel, Wlabel)
        if type == TYPE_LINEAR_PER_STCK:
            return LinearPredictorPerSegment(df, Xlabels, StckLabel,Ylabel, Wlabel)
        if type == TYPE_LSUM_OF_FUNCTIONS_PER_MKT:
            return LsumOfFunctionsPredictorPerSegment(df, Xlabels, MktLabel, Ylabel, Wlabel)
        if type == TYPE_LSUM_OF_FUNCTIONS_PER_STCK:
            return LsumOfFunctionsPredictorPerSegment(df, Xlabels, StckLabel,Ylabel, Wlabel)
        assert 0, "Predictor type unknown: " + type

    @staticmethod
    def addRegressionData(predictorType, df, addRegressionVectors = False, addRest = False):
        myPredictor = mktPredictor.predictorFactory(predictorType, df, utils.HEADER_X, utils.HEADER_Y, utils.HEADER_WEIGHT, utils.HEADER_MARKET, utils.HEADER_STOCK)
        if addRegressionVectors:
            df = df.join(pd.DataFrame(data=myPredictor.getX_reduced(), columns=['x' + str(x) + predictorType for x in range(myPredictor.getk())]))
        myPredictor.estimate(df, utils.HEADER_X, utils.HEADER_Y_PREDICT + predictorType)
        if addRest:
            df[utils.HEADER_Y_REST + predictorType] = df[utils.HEADER_Y] - df[utils.HEADER_Y_PREDICT + predictorType]
        return (myPredictor.getAvgCoeffDetermin(), df)

##############################################################################################################################
##############################################################################################################################
############################                     different types of predictors                    ############################
##############################################################################################################################
##############################################################################################################################

# PCA on a vector
# getAccurateLinearPredictor determines the optimal dimension
class LinearPredictor(mktPredictor):
    def __init__(self, X, Y, W, k):
        self.k = k
        self.pca = PCA()
        # normalize -> implies prediction should be adapted
        # there are a few arguments to not normalize but it gives better results
        # need to keep a reference on standard scaler to transform input for prediction
        self.scaler = StandardScaler(with_mean = True, with_std = False).fit(X) 
        X_std = self.scaler.transform(X)
        self.sampleSize = len(X)
        # fit the PCA
        self.X_reduced = self.pca.fit_transform(X_std)
        self.Y = Y
        self.W = W
        self.var_exp = np.round(self.pca.explained_variance_ratio_, decimals=4) * 100
        self.regr = LinearRegression()
        self.regr.fit(self.X_reduced[:,:self.k], Y, W)

    def getk(self):
        return self.k
    def setk(self, k):
        self.k = k
        # change the meaningful dimension requires re-calibrate
        self.regr = LinearRegression()
        self.regr.fit(self.X_reduced[:,:self.k], self.Y, self.W)
    def getAvgCoeffDetermin(self):
        return self.avgCoeffDetermin
    def getX_reduced(self):
        return self.X_reduced[:,:self.k]
    def setAvgCoeffDetermin(self, avgCoeffDetermin):
        self.avgCoeffDetermin = avgCoeffDetermin
    def getAvgWeightedSquareDifference(self):
        return self.avgWeightedSquareDifference
    def setAvgWeightedSquareDifference(self, avgWeightedSquareDifference):
        self.avgWeightedSquareDifference = avgWeightedSquareDifference

    @staticmethod
    def getAccurateLinearPredictor(df, Xlabels, Ylabel, Wlabel, displayCharts):
        PCAavgCoeffDeterminPercentageThreshold = 0.005
        weightedVariance = sum(np.square(df[Ylabel] -  np.average(df[Ylabel])) * df[Wlabel]) # could be diveded by sum(df[Wlabel]) to normalise but will cancel out later
        avgCoeffDetermin = []
        coeffDetermin = [[] for k in range(len(Xlabels))]
        for train, test in utils.getTestAndTrainSample().split(df[Xlabels]):
            # initialise the PCA for max dimension
            predictor = LinearPredictor(df[Xlabels].loc[train,:].values, df[Ylabel].loc[train].values, df[Wlabel].loc[train].values, len(Xlabels))
            for k in range(len(Xlabels)):
                # restrict the dimension of the predictor and calculate the avg of Coefficient of Determination
                predictor.setk(k + 1)
                # careful, there is a hack here: passing df[Xlabels].loc[test,:] as df sends a copy, so when the estimate function allocates the Y_TEST col, nothing actually affects df
                coeffDetermin[k] += [1 - sum(np.square(df.loc[test, Ylabel] - predictor.estimate(df.loc[test,Xlabels], Xlabels, utils.HEADER_Y_TEST)) * df[Wlabel].loc[test]) / weightedVariance]
        # transforming coefficient of Determination for each traning set into avgCoeffDetermins
        avgCoeffDetermin += [sum(coeffDetermin[k]) / len(coeffDetermin[k]) for k in range(len(Xlabels))]

        # choosing the predictor
        k = 0
        for i in range(len(avgCoeffDetermin) - 1, 1, -1):
            # going from all dimensions to 1, returning the first dimension which reduces avgCoeffDetermin by more than threshold
            # with threshold 2%, and 11 dimensions we lose 20% of avgCoeffDetermin in the worst case
            if (k == 0) and ((avgCoeffDetermin[i] - avgCoeffDetermin[i - 1])  > PCAavgCoeffDeterminPercentageThreshold):
                k = i

        # test that regression does not fail on a 'King-Kong' value
        if (avgCoeffDetermin[0] > 1) or (avgCoeffDetermin[0] < 0):
            raise NameError('Regression score for k = 1 gives score worst than data variance, data is not fit for regression.')
        finalPredictor = LinearPredictor(df[Xlabels], df[Ylabel], df[Wlabel], k + 1)
        finalPredictor.setAvgCoeffDetermin(avgCoeffDetermin[k])
        finalPredictor.setAvgWeightedSquareDifference((1- avgCoeffDetermin[k]) * weightedVariance)

        if displayCharts:
            print(avgCoeffDetermin)
            fig, ax = plt.subplots()
            ax.plot(np.arange(1,len(Xlabels)), avgCoeffDetermin[1:len(Xlabels)], 'b-v')
            ax.plot(k,avgCoeffDetermin[k], 'ro')
            ax.set_title('Regression coeff. for increasing PCA dimension')
            ax.set_xlabel('Number of principal components in regression')
            ax.set_ylabel('Average Coefficient of Determination')
            ax.set_xlim((-0.2,len(Xlabels) + 0.2))
            
        return finalPredictor

    def estimate(self, dfToPredict, Xlabels, Ylabel):
        dfToPredict[Ylabel] = self.estimateRO(dfToPredict, Xlabels, Ylabel)
        return dfToPredict[Ylabel].as_matrix()
    def estimateRO(self, dfToPredict, Xlabels, Ylabel):
        try:
            transformedX = self.pca.transform(self.scaler.transform(dfToPredict[Xlabels]))
        except ValueError as err:
            logging.getLogger(utils.Constants().loggerName).log(logging.INFO, err)
            logging.getLogger(utils.Constants().loggerName).log(logging.INFO, 'Vector below cannot be projected onto PCA vectors.')
            logging.getLogger(utils.Constants().loggerName).log(logging.INFO, dfToPredict[Xlabels])
            return np.full((dfToPredict[Xlabels[0]].count(), 1), np.inf)
        else:
            return self.estimateROTransformedX(transformedX[:,:self.k])
    def estimateROTransformedX(self, X):
        return self.regr.predict(X)

# transforms the data for each vector using a set of functions and tries linear interpolations on the result
# works on vectors one by one, assuming the transformation is independant
class LsumOfFunctionsPredictor(mktPredictor):

    __functions = [ArrayFunctions.identity, np.exp, ArrayFunctions.log, np.square, np.sqrt, 
                   ArrayFunctions.digit80Percent, ArrayFunctions.digit90Percent, 
                   ArrayFunctions.addRatio, ArrayFunctions.multRatio, ArrayFunctions.expOfMultRatio, ArrayFunctions.logOfMultRatio, 
                   ArrayFunctions.digit80PercentOfAddRatio, ArrayFunctions.digit90PercentOfAddRatio, ArrayFunctions.digit80PercentOfMultRatio, ArrayFunctions.digit90PercentOfMultRatio]
    @staticmethod
    def getFunctions():
        return LsumOfFunctionsPredictor.__functions
    @staticmethod
    def getFunctionListForOneFunctionOneLabel(Xlabels, oneLabel, oneFunction):
        return [ArrayFunctions.identity if x != oneLabel else oneFunction for x in Xlabels]
    @staticmethod
    def getFunctionList(Xlabels, prevFunctions, oneFunction):
        return prevFunctions + [oneFunction] + [ArrayFunctions.identity for i in range(len(Xlabels) - 1 - len(prevFunctions))]

    @staticmethod
    def testFitFunction(df, Xlabels, Xlabel, function, Ylabel, Wlabel, prevFunctions, incrementalFit):
        try:
            # two options
            if incrementalFit:
                # tries all functions given all previous ones, to incrementally improve the result. Functions start from higher correl axis so their impact should have gradually less and less impact
                predictor = LsumOfFunctionsPredictor(df, Xlabels, LsumOfFunctionsPredictor.getFunctionList(Xlabels, prevFunctions, function), Ylabel, Wlabel)
            else:
                # tries the functions one by one keepng all others to ID
                predictor = LsumOfFunctionsPredictor(df, Xlabels, LsumOfFunctionsPredictor.getFunctionListForOneFunctionOneLabel(Xlabels, Xlabel, function), Ylabel, Wlabel)
            result = (predictor.getAvgCoeffDetermin(), function)
            del predictor
        except NameError as err:
            logging.getLogger(utils.Constants().loggerName).log(logging.INFO, err)
            logging.getLogger(utils.Constants().loggerName).log(logging.INFO, function.__name__ + ' cannot be applied on the vector as it is not defined')
            result = (np.NINF, function)

        return result

    def __init__(self, df, Xlabels, XFunctions, Ylabel, Wlabel):
        dfTransformed = df.copy(deep = True)
        self.XFunctions = XFunctions
        if len(Xlabels) != len(XFunctions):
            raise NameError('LsumOfFunctionsPredictor requires a function for each of X columns.')
        for i in range(len(XFunctions)):
            np.seterr(all='raise')
            try:
                dfTransformed[Xlabels[i]] = XFunctions[i](dfTransformed[Xlabels[i]])
            except RuntimeWarning as e:
                raise NameError('X values out of that set of functions definition range. Predictor cannot be built.')
            except FloatingPointError as e:
                raise NameError('X values out of that set of functions definition range. Predictor cannot be built.')
            np.seterr(all='warn')
        # instead of testing function definition, tests if the final result contains nan or +-inf
        df_filtered = utils.testNanInDF(dfTransformed, Xlabels)
        if not df_filtered.empty:
            raise NameError('X values out of that set of functions definition range. Predictor cannot be built.')
        try:
            self.linearPredictor = LinearPredictor.getAccurateLinearPredictor(dfTransformed, Xlabels, Ylabel, Wlabel, False)
        except NameError as err:
            logging.getLogger(utils.Constants().loggerName).log(logging.INFO, err)
            raise NameError('X transformed by ' + '/'.join([f.__name__ for f in self.XFunctions]) + ' values cannot be fit by PCA.')
        self.Xlabels = Xlabels
        self.Ylabel = Ylabel
        self.df = df

    def getXFunctions(self):
        return self.XFunctions
    def getX_reduced(self):
        return self.linearPredictor.getX_reduced()
    def getAvgCoeffDetermin(self):
        return self.linearPredictor.getAvgCoeffDetermin()
    def getAvgWeightedSquareDifference(self):
        return self.linearPredictor.getAvgWeightedSquareDifference()
    def getk(self):
        return self.linearPredictor.getk()

    @staticmethod
    def getBestLsumOfFunctionsPredictor(df, Xlabels, Ylabel, Wlabel, displayCharts):
        avgCoeffDetermins = []
        bestavgCoeffDetermin = []
        XFunctionsList = []
        incrementalFit = utils.Constants().incrementalFunctionFit
        
        # tries all functions for each vector _ assumption is that vetors are independant and functions should not be cross-checked
        # a full check would have atrocious complexity
        for Xlabel in Xlabels:
            avgCoeffDeterminsForOneAxis = []
            for f in LsumOfFunctionsPredictor.getFunctions():
                avgCoeffDeterminsForOneAxis += [LsumOfFunctionsPredictor.testFitFunction(df, Xlabels, Xlabel, f, Ylabel, Wlabel, XFunctionsList, incrementalFit)]

            # once all functions are tried, take the one minimizing the avgCoeffDetermin
            sortedavgCoeffDeterminsForOneAxis = sorted(avgCoeffDeterminsForOneAxis, key=lambda x : x[0], reverse=True)
    
            XFunctionsList += [sortedavgCoeffDeterminsForOneAxis[0][1]]
            bestavgCoeffDetermin += [sortedavgCoeffDeterminsForOneAxis[0][0]]
            avgCoeffDetermins += [avgCoeffDeterminsForOneAxis]
        
        if displayCharts:
            # print chosen functions
            print([f.__name__ for f in XFunctionsList])
            print(' / '.join([f.__name__ for f in LsumOfFunctionsPredictor.getFunctions()]))
            print('\n'.join([' '.join([str(s) + ' /' for (s, f) in y]) for y in avgCoeffDetermins]))
            fig, ax = plt.subplots()
            for i, function in enumerate(LsumOfFunctionsPredictor.getFunctions()):
                ax.plot(np.arange(len(Xlabels)), [l[i][0] for l in avgCoeffDetermins], label=function.__name__, marker=utils.getShapesToPlot(i), color=((109 - 15 * i)/255, (127 - 15 * i)/255, (255 - 15 * i)/255)) # plotting avgCoeffDetermins for each function separately 
            ax.plot(np.arange(len(Xlabels)), bestavgCoeffDetermin, 'ro-')
            ax.legend(loc='upper center', shadow=True)
            plt.show()

        #returns the one giving the minimum
        return LsumOfFunctionsPredictor(df, Xlabels, XFunctionsList, Ylabel, Wlabel)

    # shortcut if a df object can be passed instead of np array
    def estimate(self, dfToPredict, Xlabels, Ylabel):
        dfToPredict[Ylabel] = 0
        return self.__estimate(dfToPredict, Xlabels, Ylabel, createCopy = False)

    def estimateRO(self, dfToPredict, Xlabels, Ylabel):
        return self.__estimate(dfToPredict, Xlabels, Ylabel, createCopy = True)

    # shortcut if a df object can be passed instead of np array
    def __estimate(self, dfToPredict, Xlabels, Ylabel, createCopy : bool):
        # pointer is only local, so original dataset is not pointing to the copy
        dfLocal = dfToPredict[[utils.HEADER_DAY, utils.HEADER_STOCK] + Xlabels + [Ylabel]].copy(deep = True)

        if len(Xlabels) != len(self.XFunctions):
            raise NameError('LsumOfFunctionsPredictor requires an X vector with the same number of columns than X used for calibration.')

        ratioFunctionHandled = False
        for i in range(len(self.XFunctions)):
            # if the function is a ratio, add records from the train sample to the ones to predict
            #       -> to be able to calculate the ratio 
            #       -> to improve accuracy
            if ArrayFunctions.isRatio(self.XFunctions[i]) and not ratioFunctionHandled:
                ratioFunctionHandled = True
                baseYValues = []
                #using a generator to filter the index that have been added already
                for rowNumber in dfLocal.index:
                    refIndex = LsumOfFunctionsPredictor.__getClosestIndex(dfLocal, rowNumber, self.df)
                    baseYValues.append((rowNumber, refIndex, self.df.loc[refIndex, self.Ylabel].copy()))
                for rowNumber, refIndex, y in baseYValues:
                    dfLocal.loc[rowNumber + 0.5, Xlabels] = self.df.loc[refIndex, Xlabels].copy()
                dfLocal.sort_index(inplace=True)
            # apply the function
            dfLocal[Xlabels[i]] = dfLocal[Xlabels[i]].apply(self.XFunctions[i])

        # computes the estimate, that will be stored in Ylabel
        self.linearPredictor.estimate(dfLocal, Xlabels, Ylabel)
        for i in range(len(self.XFunctions)):
            # if the function is a ratio, estimate gives a value relative to the neighbor
            # artifact neighbors added from the training set should be removed and predicted values adjusted
            if ArrayFunctions.isRatio(self.XFunctions[i]) and ratioFunctionHandled:
                ratioFunctionHandled = False
                for rowNumber, _, y in baseYValues:
                    dfLocal = dfLocal.drop(rowNumber + 0.5, axis=0)
                    dfLocal.loc[rowNumber, Ylabel] += y

        # return final result
        if not createCopy:
            dfToPredict[Ylabel] = dfLocal[Ylabel].as_matrix()
        return dfLocal[Ylabel].as_matrix()

    def __getClosestIndex(dfToPredict, rowNumber, df):
        if (utils.HEADER_DAY in df.columns) and (utils.HEADER_DAY in df.columns):
            day = dfToPredict.loc[rowNumber, utils.HEADER_DAY]
        else:
            day = np.NaN
        if (utils.HEADER_STOCK in df.columns) and (utils.HEADER_STOCK in df.columns):
            stock = dfToPredict.loc[rowNumber, utils.HEADER_STOCK]
        else:
            day = np.NaN

        if (not np.isnan(day)) and (not np.isnan(stock)):
            dfSerie = df.loc[df[utils.HEADER_STOCK] == stock, :]
            # no record for that stock
            if dfSerie.index.empty:
                return 0
            return np.argmax(df.loc[dfSerie.index, utils.HEADER_DAY].sort_values(ascending=False) < day)
        if (not np.isnan(day)):
            return np.argmax(df.loc[:, utils.HEADER_DAY].sort_values(ascending=False) < day)
        if (not np.isnan(stock)):
            dfSerie = df.loc[df[utils.HEADER_STOCK] == stock, :]
            # no record for that stock
            if dfSerie.index.empty:
                return 0
            return dfSerie.index[0]
        # worst case
        return 0

# parent class to factorise  predictor per market or per stock
class PredictorPerSegment(mktPredictor):
    def buildBasePredictor(subdf, Xlabels, Ylabel, Wlabel):
        raise NotImplementedError()
    # groupHeaders is the list of headers to group by
    def __init__(self, df, Xlabels, groupHeaders, Ylabel, Wlabel):
        self.segment = groupHeaders
        grouped = df.groupby(groupHeaders)
        self.basePredictors = dict()
        self.avgWeightedSquareDifference = 0
        self.XReduced = pd.DataFrame(np.full((df[Ylabel].count(), len(Xlabels)), np.inf), index = df.index.values)
        # fit a linear estimator per group
        missed = 0
        for name, subdf in grouped:
            try:
                self.basePredictors[name] = self.__class__.buildBasePredictor(subdf.reset_index(drop=True), Xlabels, Ylabel, Wlabel)
                self.avgWeightedSquareDifference += self.basePredictors[name].getAvgWeightedSquareDifference() * subdf[Ylabel].count()
                self.XReduced.loc[subdf.index, range(self.basePredictors[name].getk())] = self.basePredictors[name].getX_reduced()
            except NameError as err:
                logging.getLogger(utils.Constants().loggerName).log(logging.INFO, err)
                logging.getLogger(utils.Constants().loggerName).log(logging.INFO, 'Stock ' + str(name) + ' cannot be regressed with PCA.')
                missed += subdf[Ylabel].count()

        if df[Ylabel].count() > missed:
            self.avgWeightedSquareDifference = self.avgWeightedSquareDifference / (df[Ylabel].count() - missed)
            self.avgCoeffDetermin = 1 - self.avgWeightedSquareDifference / sum((np.square(df[Ylabel]) -  np.average(df[Ylabel]) ** 2) * df[Wlabel])

    def getAvgCoeffDetermin(self):
        return self.avgCoeffDetermin
    def getAvgWeightedSquareDifference(self):
        return self.avgWeightedSquareDifference
    def getX_reduced(self):
        return self.XReduced.loc[:, self.XReduced.columns[:self.getk()]].as_matrix()
    def getk(self):
        return max([lP.getk() for lP in self.basePredictors.values()])

    def estimate(self, dfToPredict, Xlabels, Ylabel):
        dfToPredict[Ylabel] = 0
        grouped = dfToPredict.groupby(self.segment)
        for name, subdf in grouped:
            if name in self.basePredictors.keys():
                dfToPredict.loc[dfToPredict[self.segment] == name, Ylabel] = self.basePredictors[name].estimateRO(subdf, Xlabels, Ylabel)
            else:
                dfToPredict.loc[dfToPredict[self.segment] == name, Ylabel] = np.full(np.max(subdf[Xlabels].count()), np.inf)
        return dfToPredict[Ylabel].as_matrix()

# implements the linear predictor per market or per stock
class LinearPredictorPerSegment(PredictorPerSegment):
    def buildBasePredictor(subdf, Xlabels, Ylabel, Wlabel):
        return LinearPredictor.getAccurateLinearPredictor(subdf, Xlabels, Ylabel, Wlabel, False)
# implements the linear sum of functions predictor per market or per stock
class LsumOfFunctionsPredictorPerSegment(PredictorPerSegment):
    def buildBasePredictor(subdf, Xlabels, Ylabel, Wlabel):
        return LsumOfFunctionsPredictor.getBestLsumOfFunctionsPredictor(subdf, Xlabels, Ylabel, Wlabel, False)
