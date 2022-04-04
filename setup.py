#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

setup(
    name="graphpype",
    version='0.0.8',
    packages=find_packages(),
    author="David Meunier",
    description="Graph analysis for neuropycon (using nipype, and ephypype); based on previous packages dmgraphanalysis and then dmgraphanalysis_nodes and graphpype",
    license='BSD 3',
    install_requires=['numpy',
                      'statsmodels',
                      'patsy',
                      'nipype',
                      'configparser',
                      "pandas",
                      "xlwt",
                      'networkx',
                      "matplotlib",
                      "bctpy",
                      "traits"]
)
