# PMNR-Code
This repository stores the implementation of the PMNR (Partial
Multi-Neuron Relaxation) paradigm within the SMT-based
[Marabou](https://github.com/NeuralNetworkVerification/Marabou)
verification software, including all PMNR instantiations mentioned in
the paper "Neural Network Verification using Partial Multi-Neuron
Relaxation". Furthermore, it contains all verification queries, MNIST
classifiers, running logs and final results featured in the paper.

## Benchmarks

We evaluated the PMNR paradigm on a total of 344 local robustness
queries with MNIST Classifiers including featuring the ReLU,
LeakyReLU, Max, Sign, Bilinear and Sigmoid activation functions.
Our benchmarks include 272 queries for piecewise-linear networks and
72 queries for other networks.

### MNIST Classifiers

Benchmarks Types:
- Benchmarks ```LeakyRelu5x100``` and ```LeakyRelu8x100``` are the
two benchmarks MNIST-1, MNIST-2 in the from
[VEGAS](https://github.com/ai-ar-research/vegas), which verify
robustness for the first 100 images from the MNIST test set using
epsilon value of 0.02.
- Benchmarks ```LeakyRelu14x28``` and ```ReluSignMax```,
```ReluBilinearSoftmax```, ```LeakyReluSigmoid``` verify robustness
for the first MNIST test set image using epsilon values of 0.02,
0.04, 0.06, 0.08.

Input query locations:
- ```LeakyRelu5x100```: 100 queries at ```MNIST/PiecewiseLinear/ipq/LeakyReLU5x100```.
- ```LeakyRelu8x100```: 100 queries at ```MNIST/PiecewiseLinear/ipq/LeakyRelu8x100```.
- ```LeakyRelu14x28```: 36 queries at ```MNIST/PiecewiseLinear/ipq/LeakyRelu14x28```.
- ```ReluSignMax```: 36 queries at ```MNIST/PiecewiseLinear/ipq/ReluSignMax```.
- ```ReluBilinearSoftmax```: 36 queries at ```MNIST/NonPiecewiseLinear/ipq/ReluBilinearSoftmax```.
- ```LeakyReluSigmoid```: 36 queries at ```MNIST/NonPiecewiseLinear/ipq/LeakyReluSigmoid```.

### Query Generation

Generation of all 344 input queries is handled by
```MNIST/generator.sh```, which relies on the remaining Bash and
Python scripts in ```MNIST```.

Python requirements for ```maraboupy``` and ```MNIST/generator.sh```
are listed in ```requirements.txt```.

First, ```generator.sh``` creates the directory structure of
```MNIST```. Then, it downloads the 200 input queries of the
MNIST-1, MNIST-2 benchmarks from
[VEGAS's OOPSLA22 repository](https://github.com/ai-ar-research/vegas).
Afterwards, it trains LeakyReLU 14x28, LeakyReLU+Sigmoid,
ReLU+Bilinear+Softmax, ReLU networks (stored in ```MNIST/PiecewiseLinear/onnx```, ```MNIST/NonPiecewiseLinear/onnx```),
before finally creating the remaining 144 input queries.

### Network Architectures

| Model                      | Hidden<br>Neurons | Hidden<br>Layers |                 Activations                 | Piecewise<br>Linear? |
|:--------------------------:|:-----------------:|:----------------:|:-------------------------------------------:|:--------------------:|
|   ```LeakyRelu5x100```     |        500        |        5         |              LeakyRelu&times;5              |       &check;        |
|   ```LeakyRelu8x100```     |        800        |        8         |              LeakyReLU&times;8              |       &check;        |
|   ```LeakyRelu14x28```     |        392        |        14        |              LeakyReLU&times;14             |       &check;        |
|   ```ReluSignMax```        |        511        |        6         |           ReLU&times;4, Sign, Max           |       &check;        |
|   ```ReluBilinearSoftmax```|        586        |        9         |           Relu, Bilinear, Softmax           |       &cross;        |
|   ```LeakyReluSigmoid```   |        280        |        10        | LeakyReLU&times;6,<br>Sigmoid&times;3, ReLU |       &cross;        |

## Running Marabou

### Command Structure

Once Marabou is compiled from source (see ```Marabou/README.md``` for instructions),
Marabou is executed as follows:
```
./Marabou --input-query <path-to-query.ipq> --milp-tightening=<option>
```

### Bound Tightening Algorithms

The ```milp-tightening``` option controls the algorithm used for Marabou's inital bound tightening algorithm call.
In our paper, we compared Marabou's performance with the following options:
- ```none```: Default option, uses the [DeepPoly](https://dl.acm.org/doi/10.1145/3290354) algorithm.
- ```backward-pmnr```: Uses ```PMNR```, which is the main instantiation of the PMNR paradigm from our paper.
- ```backward-pmnr-all```: Uses the ```PMNR-ALL``` instantiation of the PMNR paradigm from our paper.
- ```backward-pmnr-once```: Uses the ```PMNR (n=1)``` instantiation of the PMNR paradigm from our paper.
- ```backward-pmnr-random```: Uses the ```PMNR (Random)``` instantiation of the PMNR paradigm from our paper.
- ```backward-converge```: Uses the ```F+BC``` configuration of [Forward-Backward Abstract Interpretation](https://dl.acm.org/doi/10.1145/3563325).

## Evaluation Results

### Logs

For every bound tightening method ```<METHOD>``` out of
```DeepPoly, F+BC, PMNR, PMNR-ALL, PMNR-ONCE, PMNR-RANDOM```,
experiment logs are stored at ```Evaluation\<METHOD>```.

- From ```slurm-1.log``` until ```slurm-36```: Logs for ```ReLUSignMax``` model.
- From ```slurm-37.log``` until ```slurm-136```: Logs for ```LeakyRelu5x100``` model.
- From ```slurm-137.log``` until ```slurm-236```: Logs for ```LeakyRelu8x100``` model.
- From ```slurm-237.log``` until ```slurm-272```: Logs for ```ReluBilinearSoftmax``` model.
- From ```slurm-273.log``` until ```slurm-308```: Logs for ```LeakyReluSigmoid``` model.
- From ```slurm-309.log``` until ```slurm-344```: Logs for ```LeakyRelu14x28``` model.

### Summary Statistics

- Two ```.tex``` files contain summary statistics for the performance
of PMNR-enhanced Marabou performance compared to base Marabou and
PMNR-ALL-enhanced Marabou:
```Robustness/Results/Main-Results/Piecewise-Linear-Robustness-Summary.tex``` for piecewise-linear networks,
and ```Robustness/Results/Main-Results/Non-Piecewise-Linear-Robustness-Summary.tex``` for non-piecewise-linear networks.

- Three more ```.tex``` contain summary statistics for the performance of PMNR-enhanced compared to ```<METHOD>```-enhanced Marabou
for ```<METHOD>``` values of ```F+BC```, ```PMNR-ONCE```, ```PMNR-RANDOM```:
```Robustness/Results/Additional-Results/<METHOD>-Robustness-Summary.tex```.

### Plots

- Two ```.pdf``` files contain scatter plots which
compare the running time of PMNR-enhanced Marabou versus ```<METHOD>```-enhanced Marabou per each query
for ```<METHOD>``` values of ```DeepPoly```, ```PMNR-ALL```:
```Robustness/Results/Main-Results/<METHOD>-Robustness-Scatter.pdf```.

- Two ```.pdf``` files contain cactus plots which
compare the cumulative running time of PMNR-enhanced Marabou and number of solved queries, versus base Marabou and PMNR-ALL-enhanced Marabou:
```Robustness/Results/Main-Results/LeakyReLU-Robustness-Cactus.pdf``` for piecewise-linear networks,
and ```Robustness/Results/Main-Results/Other-Robustness-Cactus.pdf``` for non-piecewise-linear networks.

- Six more ```.pdf``` files contain cactus plots
which compare the cumulative running time per number of solved queries of PMNR-enhanced Marabou versus ```<METHOD>```-enhanced Marabou
for ```<METHOD>``` values of ```F+BC```, ```PMNR-ONCE```, ```PMNR-RANDOM```:
```Robustness/Results/Additional-Results/<METHOD>-LeakyReLU-Robustness-Cactus.pdf``` for piecewise-linear networks,
and ```Robustness/Results/Additional-Results/<METHOD>-Other-Robustness-Cactus.pdf``` for non-piecewise-linear networks.
