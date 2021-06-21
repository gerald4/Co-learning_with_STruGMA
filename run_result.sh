#!/bin/bash







for iter in 1 2 3 4 5 6 7 8 9 ; 
do 

 python3 black_box_only.py  --dataset_name german_credit --fold $iter
 python3 black_box_only.py  --dataset_name waveform --fold $iter
 python3 black_box_only.py  --dataset_name wine --fold $iter
 python3 black_box_only.py  --dataset_name bank_marketing --fold $iter
 python3 black_box_only.py  --dataset_name pima_indian_diabetes --fold $iter
 python3 black_box_only_bis.py  --dataset_name magic_gamma --fold $iter
 python3 black_box_only_bis.py  --dataset_name ionosphere --fold $iter

#Note that the lambda givedn here only helps in the clipping function. It is not the one that balances the two losses. That one is adapted using gradients.
# The number of components have been already cross-validated here.
 python3 result_cotraining.py --_lambda 10 --dataset_name pima_indian_diabetes --n_components 3 --fold $iter
 python3 result_cotraining.py --_lambda 10 --dataset_name waveform --n_components 3 --fold $iter
 python3 result_cotraining.py --_lambda 2 --dataset_name bank_marketing --n_components 3 --fold $iter 
 python3 result_cotraining.py --_lambda 2 --dataset_name german_credit --n_components 2 --fold $iter 
 python3 result_cotraining.py --_lambda 2 --dataset_name wine --n_components 3 --fold $iter
 python3 result_cotraining_bis.py --_lambda 5 --dataset_name magic_gamma --n_components 3 --fold $iter
 python3 result_cotraining_bis.py --_lambda 5 --dataset_name ionosphere --n_components 2 --fold $iter 

 done 






# echo "Finished" 
