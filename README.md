This library contains an experimental implementation of Banzhaf values for interpretation of tree-based models based on [Improved Feature Importance Computations for Tree Models: Shapley vs. Banzhaf](https://arxiv.org/abs/2108.04126). This is an alternative to [shap](https://github.com/slundberg/shap).
The main advantages over shap:
 - faster running times - even order of magnitude for some datasets.
 - better numerical stability. For very large trees (over 50) shap might give unreliable results. We observed some issues for much smaller trees of size 10.

 For majority of the cases, shap and ban give identical or very similar ordering and relative values of features. For more details please check this [paper](https://arxiv.org/abs/2108.04126).

# Authors
Adam Karczmarz & Piotr Wygocki

# Code
The computation of Banzhaf and Shapley values is implemented in C++.
See shap_banzhaf.cpp for general usage.
The sample usage for xgboost can be found in ipython/ban_xgboost.ipynb


# Prerequisites
Python 3.7.8
The list of packages is in requirements.txt
clang version 6.0.0-1ubuntu2


# Build:
    mkdir build
    cd build
    rm -rf *; cmake -D CMAKE_CXX_COMPILER=clang++ -D CMAKE_BUILD_TYPE=Release -D CMAKE_VERBOSE_MAKEFILE=true .. && make
remember about fetching submodules (see Remarks section)!

# Usage
Run
./shap_banzhaf
to get the usage.

In the most common usage (compute Banzhaf - the fastest version) one run
./shap_banzhaf f bst_boston.file boston.csv
where:

bst_boston.file contains trees in xgboost txt format
boston.csv is the csv containing the data points
The results will be stored in data/boston/bst_boston.file.banzhaf_fast

# Datasets
The smaller datasets are included in the repository.
To run experiments one needs to download the largest dataset Flights.
Links:
- boston https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
- health insurance https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction?select=train.csv
- flights https://www.kaggle.com/abdurrehmankhalid/delayedflights
- nhanes https://github.com/suinleelab/treeexplainer-study


# Remark
The shap directory contains a copy of the public repository https://github.com/slundberg/shap/.
We use the (slightly adjusted) C implementation of TREESHAP_PATH algorithm from that repository
(from the file shap/cext/tree_shap.h) as the ``shap_orig'' implementation. It is added as submodule. In order to download it, run:

    git submodule udate --init

# Licence

The code is under [MIT License](LICENSE.txt).



