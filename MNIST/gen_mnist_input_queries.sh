#!/usr/bin/env bash

save_dir=$1
onnx_file=$2
benchmark_name=$3
add_max_sign_layers=$4

# Add sign, max layers to it and form robustness queries.
mnist_test_image_indices=(0)
mnist_test_image_labels=(7 2 1)
possible_targets=(0 1 2 3 4 5 6 7 8 9)
epsilons=(0.020 0.040 0.060 0.080)

total_queries=$((${#mnist_test_image_indices[@]}*${#epsilons[@]}*(${#possible_targets[@]}-1)))
generated_queries=1
for i in ${mnist_test_image_indices[*]}
do
    for t in ${possible_targets[*]}
    do
        if [ ${t} -ne ${mnist_test_image_labels[$i]} ]
        then
	    for e in ${epsilons[*]}
	    do
	        printf "Generating ${benchmark_name} input query ${generated_queries}/${total_queries} ($((((${generated_queries}*100/${total_queries}*100)/100)))%%).\n"
			./generate_queries.py --file ${onnx_file} -e $e -i $i -t $t -s ${save_dir} --max-sign-constr=${add_max_sign_layers}
			((generated_queries++))
			printf "\n"
	    done
	fi
   done
done

# Manually correct ReLUSignMax input queries.
if [ ${add_max_sign_layers} = "True" ]
then
    ./correct_relu_sign_max_queries.py $save_dir
fi