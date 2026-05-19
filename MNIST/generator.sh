#!/usr/bin/env bash

# Creating directory structure.
leaky_relu_5x100_ipq_save_dir="PiecewiseLinear/ipq/LeakyReLU5x100"
leaky_relu_8x100_ipq_save_dir="PiecewiseLinear/ipq/LeakyReLU8x100"
leaky_relu_14x28_ipq_save_dir="PiecewiseLinear/ipq/LeakyReLU14x28"
leaky_relu_sigmoid_ipq_save_dir="NonPiecewiseLinear/ipq/LeakyReLUSigmoid"
relu_bilinear_softmax_ipq_save_dir="NonPiecewiseLinear/ipq/ReLUBilinearSoftmax"
relu_sign_max_ipq_save_dir="PiecewiseLinear/ipq/ReLUSignMax"
pl_onnx_save_dir="PiecewiseLinear/onnx"
npl_onnx_save_dir="NonPiecewiseLinear/onnx"
mkdir -p $leaky_relu_5x100_ipq_save_dir $leaky_relu_8x100_ipq_save_dir $leaky_relu_14x28_ipq_save_dir $leaky_relu_sigmoid_ipq_save_dir
mkdir -p $relu_bilinear_softmax_ipq_save_dir $relu_sign_max_ipq_save_dir $pl_onnx_save_dir $npl_onnx_save_dir

# Download LeakyReLU 5x100 and LeakyReLU 8x100 benchmarks from VEGAS's OOPSLA22 repository.
printf "\nDownload LeakyReLU5x100 and LeakyReLU8x100 benchmarks...\n"
./download_leaky_relu_5x100_8x100_queries.sh $leaky_relu_5x100_ipq_save_dir $leaky_relu_8x100_ipq_save_dir

#Generate LeakyReLU 14x28, LeakyReLUSigmoid, ReLUBilinearSoftmax onnx networks, and a ReLU onnx network for ReLUSignMax.
printf "\nTraining LeakyReLU14x28 MNIST Classifier...\n"
./gen_mnist_onnx_networks.py --architecture="leaky_relu" --epochs=33 --directory=$pl_onnx_save_dir
printf "\nTraining LeakyReLUSigmoid MNIST Classifier...\n"
./gen_mnist_onnx_networks.py --architecture="leaky_relu_sigmoid" --epochs=55 --directory=$npl_onnx_save_dir
printf "\nTraining ReLUBilinearSoftmax MNIST Classifier...\n"
./gen_mnist_onnx_networks.py --architecture="relu_bilinear_softmax" --epochs=33 --directory=$npl_onnx_save_dir
printf "\nTraining a ReLU network for ReLUSignMax MNIST Classifier...\n"
./gen_mnist_onnx_networks.py --architecture="relu" --epochs=33 --directory=$pl_onnx_save_dir

# Generate input queries for LeakyReLU 14x28, LeakyReLUSigmoid, ReLUBilinearSoftmax
# benchmarks, while adding Sign+Max layers to ReLUSignMax network.
leaky_relu_14x28_onnx_file="${pl_onnx_save_dir}/mnist_leaky_relu.onnx"
leaky_relu_sigmoid_onnx_file="${npl_onnx_save_dir}/mnist_leaky_relu_sigmoid.onnx"
relu_bilinear_softmax_onnx_file="${npl_onnx_save_dir}/mnist_relu_bilinear_softmax.onnx"
relu_sign_max_onnx_file="${pl_onnx_save_dir}/mnist_relu.onnx"

printf "\nCreating input queries for LeakyReLU14x28 MNIST Benchmarks...\n"
./gen_mnist_input_queries.sh $leaky_relu_14x28_ipq_save_dir $leaky_relu_14x28_onnx_file "LeakyReLU14x28" "False"
printf "\nCreating input queries for LeakyReLUSigmoid MNIST Benchmarks...\n"
./gen_mnist_input_queries.sh $leaky_relu_sigmoid_ipq_save_dir $leaky_relu_sigmoid_onnx_file "LeakyReLUSigmoid" "False"
printf "\nCreating input queries for ReLUBilinearSoftmax MNIST Benchmarks...\n"
./gen_mnist_input_queries.sh $relu_bilinear_softmax_ipq_save_dir $relu_bilinear_softmax_onnx_file "ReLUBilinearSoftmax" "False"
printf "\nCreating input queries for ReLUSignMax MNIST Benchmarks...\n"
./gen_mnist_input_queries.sh $relu_sign_max_ipq_save_dir $relu_sign_max_onnx_file "ReLUSignMax" "True"
