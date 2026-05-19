#! /usr/bin/env python3
'''

====================

Top contributors (to current version):
  - Haoze Andrew Wu
  - Ido Shmuel

This file is part of the Marabou project.
Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
in the top-level source directory) and their institutional affiliations.
All rights reserved. See the file COPYING in the top-level source
directory for licensing information.
'''

import argparse
import sys

from maraboupy import Marabou
from maraboupy import MarabouCore
#from maraboupy import MarabouUtils
from torchvision import datasets
from torchvision import transforms

SCALE = 1

def main():
    args, _ = arguments().parse_known_args()
    query = createQuery(args)
    if query is None:
        print("Unable to create an input query!")
        print("To define the benchmark, provide an input query / ONNX file, an epsilon (-e),",
            "target label (-t), and the index of the point in the MNIST dataset (-i).")
        sys.exit(1)

    name = ""
    if args.epsilon < 5:
        name = f"{args.save_dir}/input_query_ind{args.index}_target{args.target_label}"+\
        f"_eps{args.epsilon:.3f}.ipq"
    else:
        name = f"{args.save_dir}/input_query_ind{args.index}_target{args.target_label}"+\
        f"_eps{int(args.epsilon)}.ipq"
    MarabouCore.saveQuery(query, name)

def createQuery(args):
    suffix = (args.file).split('.')[-1]
    if suffix == "ipq":
        query = Marabou.load_query(args.file)
        encode_mnist_linf(query,
                      args.index,
                      args.epsilon,
                      args.target_label,
                      args.max_sign_constr)
        return query
    if suffix == "onnx":
        network = Marabou.read_onnx(args.file)
        query = network.getInputQuery()
        encode_mnist_linf(query,
                      args.index,
                      args.epsilon,
                      args.target_label,
                      args.max_sign_constr)
        return query
    return None

def encode_mnist_linf(query, index, epsilon, target_label, max_sign_constr):
    mnist_test = datasets.MNIST('datasets',
                                train=False,
                                download=True,
                                transform=transforms.ToTensor())
    point = SCALE * mnist_test[index][0].unsqueeze(0).numpy().flatten()
    input_vars = [query.inputVariableByIndex(i) for i in range(query.getNumInputVariables())]
    output_vars = [query.outputVariableByIndex(i) for i in range(query.getNumOutputVariables())]
    new_output_vars = []

    for x in input_vars:
        query.setLowerBound(x, max(0, point[x] - epsilon))
        query.setUpperBound(x, min(SCALE, point[x] + epsilon))
    if target_label == -1:
        print("No output constraint!")
        return

    if max_sign_constr:
        num = query.getNumberOfVariables()
        out_num = query.getNumOutputVariables()

        # Add maximum constraint on all original outputs.
        query.setNumberOfVariables(num + 2 * out_num + 1)
        var_max = num
        query.setLowerBound(var_max, -10000)
        query.setUpperBound(var_max, 10000)
        MarabouCore.addMaxConstraint(query, set(output_vars), var_max)

        # Add weighted sum layer with residual connection. # of neurons in WS layer is out_num.
        # Each neuron equals an output variable minus the maximal original output variable.
        for idx in range(out_num):
            scalar = 0
            coeffs = [1, 1, -1]
            var = output_vars[idx]
            var_max = num
            var_ws = idx + num + 1
            var_list = [var_ws, var_max, var]
            query.setLowerBound(var_ws, 0)
            query.setUpperBound(var_ws, 1)
            e = MarabouCore.Equation(MarabouCore.Equation.EquationType(0)) # 'EQ'
            for j, var in enumerate(var_list):
                e.addAddend(coeffs[j], var)
                e.setScalar(scalar)
            query.addEquation(e)

        # Replace old output layer with sign layer.
        # Output variable matching original output variable receives 1 if
        # the original is the maximal output variable, else it equals -1.
        for idx in range(out_num):
            var_ws = idx + num + 1
            idx_sign = idx
            var_sign = idx + num + out_num + 1
            query.setLowerBound(var_sign, -1)
            query.setUpperBound(var_sign, 1)
            query.markOutputVariable(var_sign, idx_sign)
            MarabouCore.addSignConstraint(query, var_ws, var_sign)
            new_output_vars.append(var_sign)

    else:
        new_output_vars = output_vars

    for i in range(10):
        if i != target_label:
            var_list = [new_output_vars[target_label], new_output_vars[i]]
            coeffs = [1, -1]
            scalar = 0
            e = MarabouCore.Equation(MarabouCore.Equation.EquationType(1)) # 'GE'
            for j, var in enumerate(var_list):
                e.addAddend(coeffs[j], var)
                e.setScalar(scalar)
            query.addEquation(e)

def arguments():
    ################################ Arguments parsing ##############################
    parser = argparse.ArgumentParser(description="Script which modifies the input/output bounds " +
                                                 "of an MNIST inputQuery/ONNX file, in order to " +
                                                 "verify L_infty robustness around some other " +
                                                 "input range.")
    parser.add_argument('-f', '--file', type=str, default=None,
                        help='The input query / ONNX file name')
    parser.add_argument('-e', '--epsilon', type=float, default=0,
                        help='The epsilon for L_infinity perturbation')
    parser.add_argument('-t', '--target-label', type=int, default=-1,
                        help='The target of the adversarial attack')
    parser.add_argument('-i,', '--index', type=int, default=0,
                        help='The index of the point in the MNIST dataset')
    parser.add_argument('--max-sign-constr', type=bool, default=False,
                        help='Add max and sign constraints for output vars')
    parser.add_argument('-s', '--save-dir', type=str, default='.',
                        help='Save directory')

    return parser

if __name__ == "__main__":
    main()
