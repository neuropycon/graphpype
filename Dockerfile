FROM ubuntu:latest
MAINTAINER David Meunier "david.meunier@univ-amu.fr"
RUN apt-get update
RUN apt-get install -y git python3-pip libpng-dev libfreetype6-dev libxft-dev libblas-dev liblapack-dev libatlas-base-dev gfortran libxml2-dev libxslt1-dev wget
#RUN apt-get install -y python3-tk

#RUN apt-get install libx11-6 libxext6 libxt6 # matlab
RUN pip3 install xvfbwrapper psutil numpy scipy matplotlib statsmodels pandas networkx==1.9 
RUN pip3 install mock prov click funcsigs pydotplus pydot rdflib pbr nibabel packaging pytest nipype
RUN mkdir -p /opt/packages/

########## radatools
WORKDIR /opt/packages/
RUN wget https://webs-deim.urv.cat/~sergio.gomez/download.php?f=radatools-4.0-linux64.tar.gz
#https://webs-deim.urv.cat/~sergio.gomez/download.php?f=radatools-5.2-linux64.tar.gz
RUN tar -xvf download.php\?f\=radatools-4.0-linux64.tar.gz

#ENV DISPLAY :0
# 
# ######### ephypype
# WORKDIR /opt/packages/
# RUN git clone https://github.com/davidmeunier79/ephypype.git
# WORKDIR /opt/packages/ephypype
# RUN python setup.py develop


################### NiftiReg
RUN wget https://sourceforge.net/projects/niftyreg/files/nifty_reg-1.3.9/NiftyReg-1.3.9-Linux-x86_64-Release.tar.gz/download
RUN tar -xvf download

################### Radatools
ENV RADA_PATH=/opt/packages/radatools-4.0-linux64
ENV PATH=$PATH:$RADA_PATH/Network_Tools
ENV PATH=$PATH:$RADA_PATH/Network_Properties
ENV PATH=$PATH:$RADA_PATH/Communities_Detection
ENV PATH=$PATH:$RADA_PATH/Communities_Tools

ENV NIFTYREG_INSTALL=/opt/packages/NiftyReg-1.3.9-Linux-x86_64-Release
ENV PATH=${PATH}:${NIFTYREG_INSTALL}/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NIFTYREG_INSTALL}/lib

#
# ########## nipype
# RUN mkdir -p /opt/packages/
#
# WORKDIR /opt/packages/
# RUN git clone https://github.com/davidmeunier79/nipype.git
#
# WORKDIR /opt/packages/nipype
# RUN python3 setup.py develop
#

########### graphpype
WORKDIR /opt/packages/
RUN git clone https://github.com/davidmeunier79/graphpype.git

WORKDIR /opt/packages/graphpype
RUN git checkout Dockerhub_version_0.0.8 # to be modified

RUN python3 setup.py develop


RUN echo $(which python) && \
    echo $(which python3) && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN python -c "import sys; print(sys.version)"

RUN python -c "import graphpype; print(graphpype.__path__)"

RUN python -c "import graphpype; print(graphpype.__version__)"

################################################## Finishing
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN rm -rf \
     /tmp/hsperfdata* \
     /var/*/apt/*/partial \
     /var/log/apt/term*
