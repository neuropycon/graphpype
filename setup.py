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
    lisence='BSD 3',
    install_requires=['numpy>=1.3.0',
                      'statsmodels==0.8.0',
                      'patsy==0.5.1',
                      'nipype',
                      'configparser',
                      "pandas==0.23.4",
                      "xlwt",
                      'networkx==1.9',
                      "matplotlib==2.2.3",
                      "bctpy",
                      "traits==5.2.0"]
)
