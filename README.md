# neural-network-cpp
Neural network built from scratch in C++ (XOR experiments)
# Neural Network from Scratch in C++

A neural network written in C++ without external libraries, learning the AND and XOR functions. Built as a first-year engineering student project to understand how weights, activation functions, and backpropagation work.

## Files
- neurotest1.cpp: single neuron learning AND
- xor.cpp: network with one hidden layer learning XOR
- experiments.cpp: compares different network configurations

## How to run
g++ experiments.cpp -o experiments
./experiments        (Windows: experiments.exe)

## Results (random seed 42)

| Hidden neurons | Learning rate | Epochs | Final error | Correct | Time |
|---|---|---|---|---|---|
| 4 | 0.5 | 20000 | 0.000505698 | 4/4 | 64.9 ms |
| 2 | 0.5 | 20000 | 0.516756 | 3/4 | 30.8 ms |
| 1 | 0.5 | 20000 | 0.683956 | 3/4 | 13.1 ms |
| 8 | 0.5 | 20000 | 0.000399636 | 4/4 | 64.1 ms |
| 4 | 0.05 | 20000 | 0.00936308 | 4/4 | 32.6 ms |
| 4 | 0.5 | 2000 | 0.00913434 | 4/4 | 3.4 ms |

## What I learned
- A single neuron cannot learn XOR because the classes are not linearly separable. A hidden layer is needed.
- With 1 or 2 hidden neurons, the network classified only 3 of 4 inputs correctly (seed 42). 4 and 8 neurons reached 4/4.
- Going from 4 to 8 neurons gave no meaningful improvement.
- Stopping training early (a TRIZ-inspired "partial action") kept 4/4 accuracy while cutting training time from 64.9 ms to 3.4 ms.

## Limitations and next steps
- XOR is a tiny problem, and each timing is a single run, so small differences are noise.
- Next step: train on a real dataset (Iris) with a separate test set.
