# -*- coding: utf-8 -*-
"""
Profiler
"""


import mainLauncher
from time import strftime
import cProfile
import pstats

genericProfilerOutput = 'C:\\Users\\LL\\Desktop\\Work\\Machine learning\\challenge forecast markets\\log\\'
genericProfilerName = 'profiler'
genericProfilerExtension = '.txt'

def getGenericProfilerName():
    return genericProfilerOutput + genericProfilerName + genericProfilerExtension

def getTimeStampedProfilerName():
    return genericProfilerOutput + genericProfilerName + strftime('%Y%m%d%H%M%S') + genericProfilerExtension

def sequentialRunner():
    mainLauncher.sequentialRunner()

if __name__ == '__main__':
    cProfile.run('sequentialRunner()', getGenericProfilerName())
    with open(getTimeStampedProfilerName(), 'w') as f:
        p = pstats.Stats(getGenericProfilerName(), stream=f)
        p.sort_stats('cumulative').print_stats()


