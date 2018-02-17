# -*- coding: utf-8 -*-

import numpy as np

import utils
import utilsTest
import modStats


##############################################################################################################################
##############################################################################################################################
####################################################### test functions #######################################################
##############################################################################################################################
##############################################################################################################################
def testRunner():
    testFuncs = []
    testFuncs += [getAccurateLinearPredictorTEST1]
    testFuncs += [getLinearPerStockPredictorTEST0]
    testFuncs += [getBestLsumOfFunctionsPredictorTEST0, getBestLsumOfFunctionsPredictorTEST1]
    testFuncs += [getBestLsumOfFunctionsPredictorTEST2, getBestLsumOfFunctionsPredictorTEST3]
    testFuncs += [getLsumOfFunctionsPerStockPredictorTEST0, getLsumOfFunctionsPerStockPredictorTEST1]
    testFuncs += [LinearPerStockPredictorGetX_reducedTEST0]
    
    allGood = True
    plot = False

    for f in testFuncs:
        utils.Constants.tearDown()
        (testVal, msg) = f(plot)
        if not testVal:
            print(f.__name__, msg)
            allGood = False
    if allGood:
        print('All gooooooooooooooooooooood !')

def getAccurateLinearPredictorTEST1(plot : False):
    (HEADERS, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT) = utilsTest.testFromFile1()
    myLinearPredictor = modStats.mktPredictor.predictorFactory(modStats.TYPE_LINEAR, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT, utils.HEADER_MARKET, utils.HEADER_STOCK, displayCharts=plot)
    estimate = myLinearPredictor.estimate(utilsTest.forPrediction1(), HEADER_X, HEADER_Y)
    return (np.array_equal(np.around(estimate, 6), [-0.823778, -1.519227]) and 
            (round(myLinearPredictor.getAvgCoeffDetermin(), 6) == 0.941377) and 
            (myLinearPredictor.getk() == 6), 
            'LinearPredictor returns unexpected Value for prediction 1 from file 1')
def getBestLsumOfFunctionsPredictorTEST0(plot : False):
    (HEADERS, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT) = utilsTest.testFromFile0()
    myLsumOfFunctionsPredictor = modStats.mktPredictor.predictorFactory(modStats.TYPE_LSUM_OF_FUNCTIONS, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT, utils.HEADER_MARKET, utils.HEADER_STOCK, displayCharts=plot)
    estimate = myLsumOfFunctionsPredictor.estimate(utilsTest.forPrediction1(), HEADER_X, HEADER_Y)
    return (np.array_equal(np.around(estimate, 6), [0.003891, 0.004240]) and 
            (round(myLsumOfFunctionsPredictor.getAvgCoeffDetermin(), 6) == 0.909716) and 
            np.array_equal([f.__name__ for f in myLsumOfFunctionsPredictor.getXFunctions()], ['multRatio', 'logOfMultRatio', 'digit90Percent', 'digit80Percent', 'digit90Percent', 'digit90Percent', 'addRatio', 'digit90Percent', 'identity', 'digit80Percent', 'square']), 
            'LsumOfFunctionsPredictor returns unexpected Value for prediction 1 from file 0')
def getBestLsumOfFunctionsPredictorTEST1(plot : False):
    (HEADERS, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT) = utilsTest.testFromFile1()
    myLsumOfFunctionsPredictor = modStats.mktPredictor.predictorFactory(modStats.TYPE_LSUM_OF_FUNCTIONS, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT, utils.HEADER_MARKET, utils.HEADER_STOCK, displayCharts=plot)
    estimate = myLsumOfFunctionsPredictor.estimate(utilsTest.forPrediction1(), HEADER_X, HEADER_Y)
    return (np.array_equal(np.around(estimate, 6), [ 0.023071, 0.022401]) and 
            (round(myLsumOfFunctionsPredictor.getAvgCoeffDetermin(), 6) == 0.943534) and 
            np.array_equal([f.__name__ for f in myLsumOfFunctionsPredictor.getXFunctions()], ['log', 'digit90Percent', 'log', 'digit80Percent', 'digit90Percent', 'log', 'log', 'identity', 'identity', 'digit90Percent', 'log']), 
            'LsumOfFunctionsPredictor returns unexpected Value for prediction 1 from file 1')
def getBestLsumOfFunctionsPredictorTEST2(plot : False):
    (HEADERS, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT) = utilsTest.testFromFile0()
    utils.Constants().incrementalFunctionFit = True
    myLsumOfFunctionsPredictor = modStats.mktPredictor.predictorFactory(modStats.TYPE_LSUM_OF_FUNCTIONS, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT, utils.HEADER_MARKET, utils.HEADER_STOCK, displayCharts=plot)
    estimate = myLsumOfFunctionsPredictor.estimate(utilsTest.forPrediction1(), HEADER_X, HEADER_Y)
    return (np.array_equal(np.around(estimate, 6), [ -0.200589, -17.082925]) and 
            (round(myLsumOfFunctionsPredictor.getAvgCoeffDetermin(), 6) == 0.932923) and 
            np.array_equal([f.__name__ for f in myLsumOfFunctionsPredictor.getXFunctions()], ['multRatio', 'logOfMultRatio', 'digit90Percent', 'digit80Percent', 'digit90Percent', 'logOfMultRatio', 'exp', 'digit90PercentOfAddRatio', 'sqrt', 'logOfMultRatio', 'square']), 
            'LsumOfFunctionsPredictor returns unexpected Value for prediction 1 from file 0 when functions are searched incrementally')
def getBestLsumOfFunctionsPredictorTEST3(plot : False):
    (HEADERS, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT) = utilsTest.testFromFile1()
    utils.Constants().incrementalFunctionFit = True
    myLsumOfFunctionsPredictor = modStats.mktPredictor.predictorFactory(modStats.TYPE_LSUM_OF_FUNCTIONS, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT, utils.HEADER_MARKET, utils.HEADER_STOCK, displayCharts=plot)
    estimate = myLsumOfFunctionsPredictor.estimate(utilsTest.forPrediction1(), HEADER_X, HEADER_Y)
    return (np.array_equal(np.around(estimate, 6), [-0.085291, -0.900866]) and 
            (round(myLsumOfFunctionsPredictor.getAvgCoeffDetermin(), 6) == 0.945467) and 
            np.array_equal([f.__name__ for f in myLsumOfFunctionsPredictor.getXFunctions()], ['log', 'identity', 'identity', 'digit80PercentOfAddRatio', 'digit90Percent', 'logOfMultRatio', 'identity', 'identity', 'identity', 'digit90Percent', 'log']), 
            'LsumOfFunctionsPredictor returns unexpected Value for prediction 1 from file 1 when functions are searched incrementally')
def getLinearPerStockPredictorTEST0(plot : False):
    utils.Constants().fractionFullSampleForTest = 0.2
    (HEADERS, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT) = utilsTest.testFromFile726()
    myLinearPerStockPredictor = modStats.mktPredictor.predictorFactory(modStats.TYPE_LINEAR_PER_STCK, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT, utils.HEADER_MARKET, utils.HEADER_STOCK)
    estimate = myLinearPerStockPredictor.estimate(utilsTest.forPrediction1(), HEADER_X, HEADER_Y)
    return (np.array_equal(np.around(estimate, 6), [-0.850607, -1.915488]) and 
            (round(myLinearPerStockPredictor.getAvgCoeffDetermin(), 6) == 0.847778) , 
            'LinearPerStockPredictor returns unexpected Value for prediction 1 from file 726')
def getLsumOfFunctionsPerStockPredictorTEST0(plot : False):
    (HEADERS, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT) = utilsTest.testFromFile726()
    myLsumOfFunctionsPerStockPredictor = modStats.mktPredictor.predictorFactory(modStats.TYPE_LSUM_OF_FUNCTIONS_PER_STCK, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT, utils.HEADER_MARKET, utils.HEADER_STOCK)
    estimate = myLsumOfFunctionsPerStockPredictor.estimate(utilsTest.forPrediction1(), HEADER_X, HEADER_Y)
    return (np.array_equal(np.around(estimate, 6), [2.530599, 3.0229]) and 
            (round(myLsumOfFunctionsPerStockPredictor.getAvgCoeffDetermin(), 6) == 0.932669) , 
            'LinearPerStockPredictor returns unexpected Value for prediction 1 from file 726')
def getLsumOfFunctionsPerStockPredictorTEST1(plot : False):
    (HEADERS, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT) = utilsTest.testFromFile726()
    utils.Constants().incrementalFunctionFit = True
    myLsumOfFunctionsPerStockPredictor = modStats.mktPredictor.predictorFactory(modStats.TYPE_LSUM_OF_FUNCTIONS_PER_STCK, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT, utils.HEADER_MARKET, utils.HEADER_STOCK)
    estimate = myLsumOfFunctionsPerStockPredictor.estimate(utilsTest.forPrediction1(), HEADER_X, HEADER_Y)
    return (np.array_equal(np.around(estimate, 6), [-0.100358, -0.008756]) and 
            (round(myLsumOfFunctionsPerStockPredictor.getAvgCoeffDetermin(), 6) == 0.937514) , 
            'LinearPerStockPredictor returns unexpected Value for prediction 1 from file 726')
def LinearPerStockPredictorGetX_reducedTEST0(plot : False):
    utils.Constants().fractionFullSampleForTest = 0.2
    (HEADERS, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT) = utilsTest.testFromFile726()
    myLsumOfFunctionsPerStockPredictor = modStats.mktPredictor.predictorFactory(modStats.TYPE_LSUM_OF_FUNCTIONS_PER_STCK, DF, HEADER_X, HEADER_Y, HEADER_WEIGHT, utils.HEADER_MARKET, utils.HEADER_STOCK)
    X_reduced = myLsumOfFunctionsPerStockPredictor.getX_reduced()
    return ((len(X_reduced) == 249) and 
            (round(X_reduced[0, 0], 6) == -317.090575) and 
            (round(X_reduced[1, 1], 6) == -0.463664) and 
            (round(X_reduced[0, 9], 6) == 0.00056 and 
            (round(X_reduced[248, 0], 6) == 61.238643) and 
            (round(X_reduced[247, 8], 6) == -0.067226) and 
            (round(X_reduced[248, 9], 6) == -0.001823)), 'getX_reduced functions of LinearPerStockPredictor returns unexpected Value for file 726')

if __name__ == '__main__':
    testRunner()