# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import utils

##############################################################################################################################
##############################################################################################################################
######################################################## create data  ########################################################
##############################################################################################################################
##############################################################################################################################
def emulateFileContent():
    HEADERS = ['x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6', 'Y']
    vect = np.array([[0.269645, 0.534016, 0.480973, 0.130, 0.464, 0.01070, 0.02180, 0.04420, 0.221217, 0.62, 2., -0.19564],  
        [0.08109336, 1.36621841, 1.32515713, 0.01660, 0.03020, 0.05260, 0.04190, 0.04190, 0.010932, 0.03410, 3., -0.379831], 
        [0.956138, 0.46487098, 0.328756, 0.07, 0.23, 0.52, 0.108, 0.203, 0.30592, 0.04, 159.69207570, -0.293], 
        [0.045310, 0.60822217, 0.45666231, 0.02, 0.07, 0.13, 0.26, 0.70, 0.20460, 0.01, 212.16037830, -0.818], 
        [0.160539, 0.47671769, 0.20506890, 0.68, 0.264, 0.642, 0.01190, 0.02060, 0.152451, 0.34, 153.72435130, 0.06540]])
    DF = pd.DataFrame(data=vect, columns=HEADERS)
    return (DF, HEADERS)
def emulateFileContentXContainsNaN():
    HEADERS = ['x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6', 'Y']
    vect = np.array([[float('nan'), 0.534016, 0.480973, 0.130, 0.464, 0.01070, 0.02180, 0.04420, 0.221217, 0.62, 2., -0.19564],  
        [0.08109336, 1.36621841, 1.32515713, 0.01660, 0.03020, 0.05260, 0.04190, 0.04190, 0.010932, 0.03410, 3., -0.379831], 
        [0.956138, 0.46487098, 0.328756, 0.07, 0.23, 0.52, 0.108, 0.203, 0.30592, 0.04, 159.69207570, -0.293], 
        [0.045310, 0.60822217, 0.45666231, 0.02, 0.07, 0.13, 0.26, 0.70, 0.20460, 0.01, 212.16037830, -0.818], 
        [0.160539, 0.47671769, 0.20506890, 0.68, 0.264, 0.642, 0.01190, 0.02060, 0.152451, 0.34, 153.72435130, 0.06540]])
    DF = pd.DataFrame(data=vect, columns=HEADERS)
    return (DF, HEADERS)

def forPrediction1():
    HEADERS = ['Day', 'Stock', 'x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']
    vect = np.array([[16, 726, 0.03246975, 0.60223482, 0.64159202, 0.77, 0.247, 0.524, 0.01080, 0.03370, 0.111459, 0.39, 50.], 
                  [26, 726, 0.01689292, 0.62757161, 0.69901122, 0.48, 0.113, 0.270, 0.646, 0.01740, 0.121704, 0.60, 70.]])
    DF = pd.DataFrame(data=vect, columns=HEADERS)
    return DF

def testFromFile0():
    sourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\trainMkt0.csv' # small subset of mkt 1 stocks
    return testFromFile(sourceFileTrain)
def testFromFile1420():
    sourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\trainMkt1420.csv' # subset for stock 1420 only
    return testFromFile(sourceFileTrain)
def testFromFile726():
    sourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\trainMkt726.csv' # subset for stock 726 only
    return testFromFile(sourceFileTrain)
def testFromFile1():
    sourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\trainMkt1.csv' # Mkt 1 stocks selected only
    return testFromFile(sourceFileTrain)
def testFromFile2():
    sourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\trainMkt2.csv' # Mkt 2 stocks selected only
    return testFromFile(sourceFileTrain)
def testFromFile3():
    sourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\trainMkt3.csv' # Mkt 3 stocks selected only
    return testFromFile(sourceFileTrain)
def testFromFile4():
    sourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\trainMkt4.csv' # Mkt 4 stocks selected only
    return testFromFile(sourceFileTrain)
def testFromFileFull():
    sourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\train.csv' # Full train file
    return testFromFile(sourceFileTrain)
def testFromFile(sourceFileTrain):
    HEADERS = [utils.HEADER_INDEX, utils.HEADER_MARKET, utils.HEADER_DAY, utils.HEADER_STOCK] + utils.HEADER_X + [utils.HEADER_Y, utils.HEADER_WEIGHT]
    return (HEADERS, utils.parseFile(sourceFileTrain), utils.HEADER_X, utils.HEADER_Y, utils.HEADER_WEIGHT)

def testFromRawFile1():
    sourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\trainMkt1.csv' # Mkt 1 stocks selected only
    return testFromRawFile(sourceFileTrain)

def testFromRawFile(sourceFileTrain):
    HEADERS = [utils.HEADER_INDEX, utils.HEADER_MARKET, utils.HEADER_DAY, utils.HEADER_STOCK] + utils.HEADER_X + [utils.HEADER_Y, utils.HEADER_WEIGHT]
    return (pd.read_csv(sourceFileTrain, index_col=0), HEADERS, utils.HEADER_X, utils.HEADER_Y, utils.HEADER_WEIGHT)
    
##############################################################################################################################
##############################################################################################################################
####################################################### test functions #######################################################
##############################################################################################################################
##############################################################################################################################
def testRunner():
    testFuncs = []
    testFuncs += [testConstantsGetYVarianceBucketsDefault, testConstantsSetYVarianceBucketsSetTo22]
    testFuncs += [testConstantsGetFractionFullSampleForTestDefault, testConstantsGetFractionFullSampleForTestSetTo22]
    testFuncs += [testNanInDF, testNanInDFTEST1, testNanInDFTEST2]
    testFuncs += [testConstantsGetRandomStateDefault, testConstantsGetTestSamplingRepeatsDefault, testConstantsGetLoggerNameDefault]
    testFuncs += [testConstantsGetNbThreads, testConstantsGetIncrementalFunctionFitDefault, testConstantsGetIncrementalFunctionFitSetToTrue]
    testFuncs += [testTearDown]
    
    allGood = True

    for f in testFuncs:
        (testVal, msg) = f()
        if not testVal:
            print(f.__name__, msg)
            allGood = False
    if allGood:
        print('All gooooooooooooooooooooood !')

def testConstantsGetYVarianceBucketsDefault():
    return (utils.Constants().YVarianceBuckets == 10, 'Default value given by Constant for Yvariance_buckets not as expected.')
def testConstantsSetYVarianceBucketsSetTo22():
    utils.Constants().YVarianceBuckets = 22
    return (utils.Constants().YVarianceBuckets == 22, 'Constant does not return the value set for Yvariance_buckets.')
def testConstantsGetFractionFullSampleForTestDefault():
    return (utils.Constants().fractionFullSampleForTest == 0.1, 'Default value given by Constant for FractionFullSampleForTest not as expected.')
def testConstantsGetFractionFullSampleForTestSetTo22():
    utils.Constants().fractionFullSampleForTest = 0.22
    return (utils.Constants().fractionFullSampleForTest == 0.22, 'Constant does not return the value set for FractionFullSampleForTest.')
def testConstantsGetRandomStateDefault():
    return (utils.Constants().randomState == 12883823, 'Default value given by Constant for random_state not as expected.')
def testConstantsGetTestSamplingRepeatsDefault():
    return (utils.Constants().testSamplingRepeats == 20, 'Default value given by Constant for testSamplingRepeats not as expected.')
def testConstantsGetLoggerNameDefault():
    return (utils.Constants().loggerName == utils.MAIN_LOGGER_NAME, 'Default value given by Constant for loggerName not as expected.')
def testConstantsGetNbThreads():
    return (utils.Constants().nbThreads == 4, 'Default value given by Constant for nbThreads not as expected.')
def testConstantsGetIncrementalFunctionFitDefault():
    return (utils.Constants().incrementalFunctionFit == False, 'Default value given by Constant for incrementalFunctionFit not as expected.')
def testConstantsGetIncrementalFunctionFitSetToTrue():
    utils.Constants().incrementalFunctionFit = True
    return (utils.Constants().incrementalFunctionFit == True, 'Constant does not return the value set for incrementalFunctionFit.')
def testTearDown():
    utils.Constants().YVarianceBuckets = 22
    utils.Constants().fractionFullSampleForTest = 0.22
    utils.Constants().incrementalFunctionFit = True
    utils.Constants.tearDown()
    return ((utils.Constants().YVarianceBuckets == 10) and
            (utils.Constants().fractionFullSampleForTest == 0.1) and
            (utils.Constants().incrementalFunctionFit == False), 
            'Constant tear down method does lead to param reset.')
    
def testNanInDF():
    (DF, HEADERS, HEADER_X, HEADER_Y, HEADER_WEIGHT) = testFromRawFile1()
    filtered = utils.testNanInDF(DF, HEADER_X)
    return (np.array_equal(filtered.index.values, np.array([59134])), 'testNanInDF returns unexpected list of values for file 1.')
def testNanInDFTEST1():
        return (utils.testNanInDF(*emulateFileContent()).empty, 'testNanInDF finds nan where it should not')
def testNanInDFTEST2():
        return (not utils.testNanInDF(*emulateFileContentXContainsNaN()).empty, 'testNanInDF does not see nan')
    
if __name__ == '__main__':
    testRunner()