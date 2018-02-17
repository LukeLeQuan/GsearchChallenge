# -*- coding: utf-8 -*-

import numpy as np
import random
import logging

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import utils
import modStats

##############################################################################################################################
##############################################################################################################################
#######################################################  file parsing  #######################################################
##############################################################################################################################
##############################################################################################################################

genericSourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\trainMkt'
genericSourceFileExtension = '.csv'
#sourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\trainMkt1.csv' # Mkt 1 stocks selected only
#sourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\trainMkt2.csv' # Mkt 2 stocks selected only
#sourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\trainMkt3.csv' # Mkt 3 stocks selected only
#sourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\trainMkt4.csv' # Mkt 4 stocks selected only
#sourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\train.csv' # Full train file
#sourceFileTrain = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\test.csv' # Full test file

def getPCAProcessedFile(sourceFile):
    return sourceFile[:-len(genericSourceFileExtension)] + '_out' + sourceFile[-len(genericSourceFileExtension):]
def getSourceFile(fileNumber : int):
    return genericSourceFileTrain + str(fileNumber) + genericSourceFileExtension


##############################################################################################################################
##############################################################################################################################
################################################# calling linear regression  #################################################
##############################################################################################################################
##############################################################################################################################

# Parses the raw file, works PCA per market and per stock for raw X or best f(X), then prints the output in a CSV file
def processRawFileWithLinearRegressions(sourceFile, targetFile, fractionFullSampleForTest = 0.2, YVarianceBucketsParam = 10):
    utils.Constants().fractionFullSampleForTest = fractionFullSampleForTest
    utils.Constants().YVarianceBuckets = YVarianceBucketsParam
    logging.basicConfig(filename='C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\log\\PCA.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

    df = utils.parseFile(sourceFile)

    # add discretised variance in df
    utils.addYVarianceGroups(df, utils.HEADER_Y, utils.HEADER_STOCK, utils.HEADER_Y_VARIANCE, utils.HEADER_Y_VARIANCE_GROUP)
    # simple PCA on the whole serie
    coeffDetermin, df = modStats.mktPredictor.addRegressionData(modStats.TYPE_LINEAR, df, addRegressionVectors = True, addRest = False)
    print('Linear predictor per market average coefficient of determination: ' + str(coeffDetermin))
    # PCA on linear combination of functions on the whole serie
    coeffDetermin, df = modStats.mktPredictor.addRegressionData(modStats.TYPE_LSUM_OF_FUNCTIONS, df, addRegressionVectors = True, addRest = False)
    print('Linear sum of functions predictor per market average coefficient of determination: ' + str(coeffDetermin))
    # simple PCA on each stock independantly
    coeffDetermin, df = modStats.mktPredictor.addRegressionData(modStats.TYPE_LINEAR_PER_STCK, df, addRegressionVectors = True, addRest = True)
    print('Linear predictor per stock average coefficient of determination: ' + str(coeffDetermin))
    # PCA on linear combination of functions on each stock independantly 
    coeffDetermin, df = modStats.mktPredictor.addRegressionData(modStats.TYPE_LSUM_OF_FUNCTIONS_PER_STCK, df, addRegressionVectors = True, addRest = True)
    print('Linear sum of functions predictor per stock average coefficient of determination: ' + str(coeffDetermin))

    # write the new file with predicted data
    df.to_csv(targetFile)
    return df

##############################################################################################################################
##############################################################################################################################
########################################################### plots  ###########################################################
##############################################################################################################################
##############################################################################################################################

def plotPCA(df):
    # sampling
    sample=random.sample(range(np.max(df.count())), 20000)
    # display charts one by one

    XSet = [[(utils.HEADER_X[0], 'X0'), (utils.HEADER_X[1], 'X1')], 
            [('X0' + modStats.TYPE_LINEAR, 'X0 from PCA'), ('X1' + modStats.TYPE_LINEAR, 'X1 from PCA')], 
            [('X0' + modStats.TYPE_LSUM_OF_FUNCTIONS, 'X0 from PCA of LF(X)'), ('X1' + modStats.TYPE_LSUM_OF_FUNCTIONS, 'X1 from PCA of LF(X)')], 
            [('X0' + modStats.TYPE_LINEAR_PER_STCK, 'X0 from PCA per stock'), ('X1' + modStats.TYPE_LINEAR_PER_STCK, 'X1 from PCA per stock')], 
            [('X0' + modStats.TYPE_LSUM_OF_FUNCTIONS_PER_STCK, 'X0 from PCA of LF(X) per stock'), ('X1' + modStats.TYPE_LSUM_OF_FUNCTIONS_PER_STCK, 'X1 from PCA of LF(X) per stock')]]

    # original problem
    plotAllPCASetsForOneY (XSet, utils.HEADER_Y, sample, 'Y', colour = 'b')
    # rest after PCA per stock
    plotAllPCASetsForOneY (XSet, utils.HEADER_Y_REST + modStats.TYPE_LINEAR_PER_STCK, sample, 'Rest after PCA per stock', colour = 'r')
    # rest after PCA per stock
    plotAllPCASetsForOneY (XSet, utils.HEADER_Y_REST + modStats.TYPE_LSUM_OF_FUNCTIONS_PER_STCK, sample, 'Rest after PCA per stock', colour = 'g')


def plotAllPCASetsForOneY(XSet, Ylabel, sample, titleHeader, colour):
    maxGpNb = len(XSet)
    maxCol = max([len(s) for s in XSet])

    plt.figure(figsize=(15, 20))
    for gp in range(maxGpNb):
        colCounter = 0
        for colX, paramTitle in XSet[gp]:
            colCounter += 1
            
            plt.subplot(maxGpNb, maxCol, gp * maxCol + colCounter)
            plotOnePCASet(df, colX, Ylabel, titleHeader + '(' + paramTitle + ')', sample, colour = colour)

    plt.show()

def plotOnePCASet(df, colX, colY, title, sample, colour = 'b'):
    # scaling the vectors and cutting the edges (approx to visualize trends better)
    # will be scaled between 0 and 1
    x = (np.min(df.loc[sample, colX]) + df.loc[sample, colX])
    capX = percentileBoundaries(x)
    x = np.minimum(x, capX)
    x = x / np.max(x)
    # a cap / floor is applied to Y 
    capY = percentileBoundaries(df.loc[sample, colY])
    y = np.maximum(np.minimum(df.loc[sample, colY], capY), -capY)
    plt.scatter(x, y, c = colour, alpha=0.7, cmap=cm.Paired)
    plt.title(title)

def percentileBoundaries(vect):
    percentiles = [np.percentile(np.abs(vect), p) for p in [70, 80, 90, 95, 97, 98, 99, 100]]
    for i in range(len(percentiles) - 1):
        if (percentiles[i + 1] - percentiles[i]) > (5 * percentiles[i]):
            return percentiles[i]
    return percentiles[-1]
##############################################################################################################################
##############################################################################################################################
############################################################ main ############################################################
##############################################################################################################################
##############################################################################################################################

if __name__ == '__main__':
    # main parameters
    sourceFileNumber = 1
    performPCA = True # takes time, if false the code will use saved data from a previous run
    utils.Constants().incrementalFunctionFit = True # linear sum of functions will fit functions incrementally if True, independantly if False
    
    # chose the file to process
    sourceFile = getSourceFile(sourceFileNumber)
    # derive the post PCA / regression file name
    postLinearRegressionProcessingFile = getPCAProcessedFile(sourceFile)
    # gets X values post PCA and prediction for all Ys
    if performPCA:
        # calls PCA / linear regression formula to prepare different sets of X0 to process and the prediction for each method
        df = processRawFileWithLinearRegressions(sourceFile, postLinearRegressionProcessingFile)
    else:
        # recalls the last file saved post PCA
        df = utils.parseFile(postLinearRegressionProcessingFile)

    plotPCA(df)








