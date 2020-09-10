#!/bin/bash




#python3 result_cotraining.py --_lambda 20 --dataset_name ionosphere --n_components 3 --fold 0


#for data in wine magic_gamma ionosphere pima_indian_diabetes ;
#    do
#python3 result_cotraining.py --_lambda 5 --dataset_name pima_indian_diabetes --n_components 3 --fold 0

python3 result_cotraining.py --_lambda 10 --dataset_name magic_gamma --n_components 2 --fold 0

#python3 result_cotraining.py --_lambda 10 --dataset_name ionosphere --n_components 3 --fold 0
#python3 result_cotraining.py --_lambda 5 --dataset_name adult --n_components 2 --fold 0
#python3 result_cotraining.py --_lambda 2 --dataset_name bank_marketing --n_components 3 --fold 0

#python3 result_cotraining.py --_lambda 5 --dataset_name wine --n_components 2 --fold 0



echo "Finished" 
