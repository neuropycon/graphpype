FROM ubuntu:16.04
MAINTAINER David Meunier "david.meunier@inserm.fr"
RUN apt-get update
RUN apt-get install -y git python3-pip libpng-dev libfreetype6-dev libxft-dev libblas-dev liblapack-dev libatlas-base-dev gfortran libxml2-dev libxslt1-dev wget
RUN apt-get install -y python3-tk

#RUN apt-get install libx11-6 libxext6 libxt6 # matlab
RUN pip3 install xvfbwrapper psutil numpy scipy matplotlib statsmodels pandas networkx==1.9 
RUN pip3 install mock prov click funcsigs pydotplus pydot rdflib pbr nibabel packaging pytest
#nipype==0.12
RUN mkdir -p /root/packages/
########## nipype

WORKDIR /root/packages/
RUN git clone https://github.com/davidmeunier79/nipype.git
WORKDIR /root/packages/nipype
RUN python3 setup.py develop


########### graphpype 
WORKDIR /root/packages/
RUN git clone https://github.com/davidmeunier79/graphpype.git
WORKDIR /root/packages/graphpype
RUN python3 setup.py develop
RUN git checkout dev  ###

########## radatools
WORKDIR /root/packages/
RUN wget http://deim.urv.cat/~sergio.gomez/download.php?f=radatools-4.0-linux64.tar.gz
RUN tar -xvf download.php\?f\=radatools-4.0-linux64.tar.gz

#ENV DISPLAY :0
# 
# ######### ephypype
# WORKDIR /root/packages/
# RUN git clone https://github.com/davidmeunier79/ephypype.git
# WORKDIR /root/packages/ephypype
# RUN python setup.py develop


################### NiftiReg
RUN wget https://sourceforge.net/projects/niftyreg/files/nifty_reg-1.3.9/NiftyReg-1.3.9-Linux-x86_64-Release.tar.gz/download
RUN tar -xvf download


ENV RADA_PATH=/root/packages/radatools-4.0-linux64
ENV PATH=$PATH:$RADA_PATH/Network_Tools
ENV PATH=$PATH:$RADA_PATH/Network_Properties
ENV PATH=$PATH:$RADA_PATH/Communities_Detection
ENV PATH=$PATH:$RADA_PATH/Communities_Tools

ENV NIFTYREG_INSTALL=/root/packages/NiftyReg-1.3.9-Linux-x86_64-Release
ENV PATH=${PATH}:${NIFTYREG_INSTALL}/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NIFTYREG_INSTALL}/lib