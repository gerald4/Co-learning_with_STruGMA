#!/bin/bash




#python3 result_cotraining.py --_lambda 20 --dataset_name ionosphere --n_components 3 --fold 0


#for data in wine magic_gamma ionosphere pima_indian_diabetes ;
#    do
#python3 result_cotraining.py --_lambda 5 --dataset_name pima_indian_diabetes --n_components 3 --fold 0

#python3 result_cotraining.py --_lambda 10 --dataset_name magic_gamma --n_components 2 --fold 0

#python3 result_cotraining.py --_lambda 10 --dataset_name ionosphere --n_components 3 --fold 0
#python3 result_cotraining.py --_lambda 5 --dataset_name adult --n_components 2 --fold 0
#python3 result_cotraining.py --_lambda 2 --dataset_name bank_marketing --n_components 3 --fold 0

#python3 result_cotraining.py --_lambda 5 --dataset_name wine --n_components 2 --fold 0

#Toto

#python3 result_cotraining.py --_lambda 5 --dataset_name student_performance --n_components 2 --fold 0

#python3 result_cotraining.py --_lambda 10 --dataset_name wine --n_components 3 --fold 0

#python3 result_cotraining.py --_lambda 10 --dataset_name magic_gamma --n_components 3 --fold 0

#python3 result_cotraining.py --_lambda 10 --dataset_name ionosphere --n_components 3 --fold 0 # or 2 2

#python3 result_cotraining.py --_lambda 5 --dataset_name pima_indian_diabetes --n_components 3 --fold 0 # or 3 10

#python3 result_cotraining.py --_lambda 5 --dataset_name bank_marketing --n_components 2 --fold 0





for iter in 5 6 7 8 9 ; 
do #python3 result_cotraining_bis.py --_lambda 5 --dataset_name magic_gamma --n_components 3 --fold $iter

 #python3 black_box_only.py  --dataset_name german_credit --fold $iter
 #python3 black_box_only.py  --dataset_name waveform --fold $iter
 #python3 black_box_only.py  --dataset_name wine --fold $iter
  python3 black_box_only.py  --dataset_name bank_marketing --fold $iter
  python3 black_box_only.py  --dataset_name pima_indian_diabetes --fold $iter
  python3 black_box_only_bis.py  --dataset_name magic_gamma --fold $iter
 #python3 black_box_only_bis.py  --dataset_name ionosphere --fold $iter

 #python3 result_cotraining.py --_lambda 10 --dataset_name pima_indian_diabetes --n_components 3 --fold $iter
 #python3 result_cotraining.py --_lambda 10 --dataset_name waveform --n_components 3 --fold $iter
 #python3 result_cotraining.py --_lambda 2 --dataset_name bank_marketing --n_components 3 --fold $iter # 10 3, 5 3, 2 2
 #python3 result_cotraining.py --_lambda 2 --dataset_name german_credit --n_components 2 --fold $iter # 2 3 or 5 3
 #python3 result_cotraining.py --_lambda 2 --dataset_name wine --n_components 3 --fold $iter

 #python3 result_cotraining_bis.py --_lambda 5 --dataset_name ionosphere --n_components 2 --fold $iter # 5 3

 done 















# python3 result_cotraining.py --_lambda 2 --dataset_name bank_marketing --n_components 3 --fold 4 # 10 3, 5 3, 2 2
# python3 result_cotraining.py --_lambda 10 --dataset_name pima_indian_diabetes --n_components 3 --fold 4
# python3 result_cotraining.py --_lambda 5 --dataset_name ionosphere --n_components 3 --fold 4 # 5 3
# python3 result_cotraining.py --_lambda 5 --dataset_name magic_gamma --n_components 3 --fold 4 # 2 3 or 5 3
# python3 result_cotraining.py --_lambda 2 --dataset_name wine --n_components 3 --fold 4

# python3 result_cotraining.py --_lambda 2 --dataset_name german_credit --n_components 2 --fold 4





# echo "Finished" 
