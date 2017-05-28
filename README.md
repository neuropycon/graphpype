graphpype
===============

Neuropycon project for graph analysis, can be used from ephypype and nipype

Documentation
-------------

http://neuropycon.github.io/neuropycon_doc/graphpype/

Radatools
---------
You should add all the directories from radatools to the PATH env variable:

1. Download radatools sotware:

http://deim.urv.cat/~sergio.gomez/radatools.php#download

2. Download and extract the zip file

3. Add following lines in your .bashrc:

### For radatools 3.2:
RADA_PATH=/home/david/Tools/Software/radatools-3.2-linux32
(replace /home/david/Tools/Software by your path to radatools)

* export PATH=$PATH:$RADA_PATH/01-Prepare_Network/
* export PATH=$PATH:$RADA_PATH/02-Find_Communities/
* export PATH=$PATH:$RADA_PATH/03-Reformat_Results
* export PATH=$PATH:$RADA_PATH/04-Other_Tools/

### For radatools 4.0:
RADA_PATH=/home/david/Tools/Software/radatools-4.0-linux64
(replace /home/david/Tools/Software by yout path to radatools)

* export PATH=$PATH:$RADA_PATH/Network_Tools
* export PATH=$PATH:$RADA_PATH/Network_Properties
* export PATH=$PATH:$RADA_PATH/Communities_Detection 
* export PATH=$PATH:$RADA_PATH/Communities_Tools

Good practice for developpers:
------------------------------    
    1. Fork the package on your github
    
    2. clone the forked package as origin 
    git clone https://github.com/your_github_login/graphpype.git
    
    3. add neuropycon repo as upstream 
    git remote add upstream  https://github.com/neuropycon/graphpype.git
    
    4. create a new branch before modifying any part of the code, or make a fresh clone, make a branch and report your modifications. The origin checkout as to as as fresh as possible
    git checkout -b my_new_branch
    
    5. commit and the new branch to your forked version of the packages
    git commit -m"My modifications" -a 
    git pull origin my_new_branch
    
    6. make a pull request on neuropycon version of the package
    
    
Magical sentence to modify all the import neuropype_graph -> graphpype:
-----------------------------------------------------------------------
    
find dir_path -type f -print0 | xargs -0 sed -i 's/old_name/new_name/g'

* example:

find ~/Tools/python/Projects/my_project -type f 
-print0 | xargs -0 sed -i 's/neuropype_graph/graphpype/g'
