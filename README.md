#LdSM-WS

This code is a modified version the LdSM algorithm proposed in the research paper titled “LdSM: Logarithm-depth Streaming 
Multi-label Decision Trees” authored by Maryam Majzoubi and Anna Choromanska which was published at AISTATS 2020. 
The code is written in C++ and should compile on 64 bit Windows/Linux machines using a C++11 enabled compiler. 

Data sets
---------------------------------------
1. Download 500 dimension word2vec vectors from https://wikipedia2vec.github.io/wikipedia2vec/pretrained/

2. Download benchmark datasets from below repository
(http://manikvarma.org/downloads/XC/XMLRepository.html). 
The data format required for LdSM is different than the 
original data sets. In order to format the original data please use the ```format_data.ipynb``` script. Store the resulting datasets it in the folder data2.

Compiling
---------------------------------------
mkdir build && cd build && cmake .. && make

Running experiments
---------------------------------------
mkdir results && cd scripts && ./run_wiki10.sh

Original LdSM Code - https://github.com/mmajzoubi/LdSM

