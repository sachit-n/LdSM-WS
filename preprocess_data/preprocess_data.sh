#!/bin/sh

#  preprocess_data.sh
#  
#
#  Created by Sachit Nagpal on 12/5/21.
#  

dataset="wiki10_13k"
data_dir="data/$dataset"
results_dir="warm_start_data/$dataset"

frac=0.8   # $frac percentage of ground truth labels will be revealed per each test point, while rest of the points are held-out for evaluation purposes
revpct="80"

trn_file="${data_dir}/train.txt"  # training points' user features and ground truth labels
tst_file="${data_dir}/test.txt"    # test points' user features and ground truth labels
#Y_Yf_file="${data_dir}/Y_Yf.txt"    # label or item features for all the labels
#
trn_X_Xf_file="${results_dir}/train_X.txt"    # training points' user features
#trn_item_X_Xf_file="${results_dir}/trn_item_X_Xf.txt"    # training points' item-set features
trn_X_Y_file="${results_dir}/train_Y_${revpct}.txt"    # training points' ground truth labels
tst_X_Xf_file="${results_dir}/test_X.txt"    # test points' user features
#tst_item_X_Xf_file="${results_dir}/tst_item_X_Xf.txt"    # test points' item-set features
inc_tst_X_Y_file="${results_dir}/inc_test_Y_${revpct}.txt"    # test points' revealed labels
exc_tst_X_Y_file="${results_dir}/exc_test_Y_${revpct}.txt"    # test points' unrevealed, held-out labels for evaluation
#inv_prop_file="${results_dir}/inv_prop.txt"    # labels' inverse propensity scores
#score_file="${results_dir}/score_mat.txt"    # predicted label scores for the test points

#A=0.55    # 'A' parameter in the inverse propensity model. Refer to README.txt
#B=1.5    # 'B' parameter in the inverse propensity model. Refer to README.txt
#N=15539 # Number of points in $trn_file

matlab -nodesktop -nodisplay -r "cd('$PWD'); addpath(genpath('../Tools')); swiftXML_preprocess_data( '$trn_file', '$tst_file', '$trn_X_Xf_file', '$trn_X_Y_file', '$tst_X_Xf_file', '$inc_tst_X_Y_file', '$exc_tst_X_Y_file', $frac ); exit;"
