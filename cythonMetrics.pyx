# -*- coding: utf-8 -*-


cimport numpy as np
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

##############################################################################################################################
##############################################################################################################################
#########################################################   metrics  #########################################################
##############################################################################################################################

cdef double genericDistance(double[:]x, double[:]y, double weightDay, double weightStock):
    cdef double res

    if y[1] != x[1]:
        res = abs(y[0] - x[0]) + 1
    else:
        res = abs(y[0] - x[0])
    
    for i in range(2, x.shape[0]):
        res += (x[i] - y[i])**2

    return res

cdef class CustomMetrics:

    cdef double dayWeight
    cdef double stockWeight

    def __init__(self, double dayWeight, double stockWeight):
        self.dayWeight = dayWeight
        self.stockWeight = stockWeight

    cpdef double distDayStockX(self, double[:] x, double[:] y):
        return genericDistance(x, y, self.dayWeight, self.stockWeight)


##############################################################################################################################
##############################################################################################################################
########################################################  classifier  ########################################################
################################################  K nearest neighbors tuning  ################################################
##############################################################################################################################

# encapsulates the KNN setup such that the constructor idoes not require arguments
cdef class TunedKNNDistDayStockX:

    cdef CustomMetrics customMetrics
    cdef int kInKNN
    cdef object predictor

    def __init__(self, int kInKNN, double distDayWeight, double distStockWeight):
        self.customMetrics = CustomMetrics(distDayWeight, distStockWeight)
        self.kInKNN = kInKNN
        self.predictor = None

    def fit(self, np.ndarray[np.float_t, ndim=2] X, np.ndarray[np.int64_t, ndim=1] Y):
        self.predictor = KNeighborsClassifier(n_neighbors=self.kInKNN, metric=self.customMetrics.distDayStockX)
        self.predictor.fit(X, Y)

    def predict(self, np.ndarray[np.float_t, ndim=2] X):
        return self.predictor.predict(X)
