#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup
import os

setup(
    name = "graphpype",
    version = '0.0.1dev',
    packages = ['graphpype'],
    author = "David Meunier",
    description = "Graph analysis for neuropycon (using nipype, and ephypype); based on previous packages dmgraphanalysis and then dmgraphanalysis_nodes and graphpype"   , 
    lisence='BSD 3',
    install_requires=['numpy>=1.3.0',
                      'configparser',
                      "pandas",
                      "matplotlib"]
)

