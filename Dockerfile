FROM ubuntu:latest
MAINTAINER David Meunier "david.meunier@univ-amu.fr"
RUN apt-get update
RUN apt-get install -y git python3-pip libpng-dev libfreetype6-dev libxft-dev libblas-dev liblapack-dev libatlas-base-dev gfortran libxml2-dev libxslt1-dev wget
#RUN apt-get install -y python3-tk

#RUN apt-get install libx11-6 libxext6 libxt6 # matlab
RUN pip3 install xvfbwrapper psutil numpy scipy matplotlib statsmodels pandas networkx==1.9 
RUN pip3 install mock prov click funcsigs pydotplus pydot rdflib pbr nibabel packaging pytest nipype

RUN pip3 scikit-image

RUN mkdir -p /opt/packages/

#ENV DISPLAY :0
#
# ######### ephypype
# WORKDIR /opt/packages/
# RUN git clone https://github.com/davidmeunier79/ephypype.git
# WORKDIR /opt/packages/ephypype
# RUN python setup.py develop

################### Radatools
WORKDIR /opt/packages/
RUN wget https://webs-deim.urv.cat/~sergio.gomez/download.php?f=radatools-4.0-linux64.tar.gz
#https://webs-deim.urv.cat/~sergio.gomez/download.php?f=radatools-5.2-linux64.tar.gz
RUN tar -xvf download.php\?f\=radatools-4.0-linux64.tar.gz

ENV RADA_PATH=/opt/packages/radatools-4.0-linux64
ENV PATH=$PATH:$RADA_PATH/Network_Tools
ENV PATH=$PATH:$RADA_PATH/Network_Properties
ENV PATH=$PATH:$RADA_PATH/Communities_Detection
ENV PATH=$PATH:$RADA_PATH/Communities_Tools


################### NiftiReg

RUN apt-get install --yes --no-install-recommends \
        ca-certificates cmake gcc g++ git make \
    && git clone https://github.com/KCL-BMEIS/niftyreg.git /tmp/niftyreg-src \
    && mkdir /tmp/niftyreg-build \
    && cd /tmp/niftyreg-build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/niftyreg /tmp/niftyreg-src \
    && make \
    && make install

ENV PATH=/opt/niftyreg/bin:$PATH

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

ADD https://api.github.com/repos/davidmeunier79/graphpype/git/refs/heads/Dockerhub_version_0.0.8 version.json

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
