# -*- coding: utf-8 -*-

import setuptools
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="cythonMetrics", 
      ext_modules=cythonize('cythonMetrics.pyx'),
      include_dirs=[numpy.get_include()]
)