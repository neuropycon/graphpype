#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
import os

setup(
    name = "graphpype",
    version = '0.0.1dev0',
    packages = ['graphpype'],
    author = "David Meunier",
    description = "Graph analysis for neuropycon (using nipype, and ephypype); based on previous packages dmgraphanalysis and then dmgraphanalysis_nodes and graphpype"   , 
    lisence='BSD 3',
    install_requires=['numpy>=1.3.0',
                      'nipype<=0.12',
                      'configparser',
                      "pandas",
                      "matplotlib",
                      "xvfbwrapper==0.2.9"]
)

