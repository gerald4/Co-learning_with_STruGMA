#!/bin/bash





#for data in  magic_gamma ionosphere pima_indian_diabetes ;
#    do
#student_performance wine 
for data in  german_credit wine magic_gamma ionosphere pima_indian_diabetes bank_marketing;
	do 
	for K in 2 3 4;
	    do
		for lambda in 2 5 10;
		do
		python3 co_training_evaluation.py --_lambda $lambda --dataset_name $data --n_components $K --type ""
		done
	done
done

#for data in adult ; #indian_liver;
#	do 
#	for K in 2 3;
#	    do
#		for lambda in 2 5 10;
#		do
#		python3 co_training_evaluation.py --_lambda $lambda --dataset_name $data --n_components $K --type ""
#		done
#	done
#done
#    done



echo "Finished" 
