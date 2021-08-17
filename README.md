# Qroestl
A thin optimization layer on top of Qiskit. Named after the delicious Austrian & Bavarian dish [Gröstl](https://de.wikipedia.org/wiki/Tiroler_Gröstl) which I ate when I started this ;-)

## Install

First clone the repo:
```bash
$ git clone https://github.com/ssenge/Qroestl.git
```

Then create an environment, e.g. using _conda_:
```bash
$ conda create -n qc  python=3.8
```

(Note that it seems as the latest version of CPLEX does not work with Python 3.9, so use 3.8 if you want to use CPLEX.)

Activate the environment:

```bash
$ conda activate qc
```

Install requirements:
```bash
$ pip install -r requirements.txt
```

Optionally, if you want to use CPLEX install it according to the official [documentation](https://www.ibm.com/products/ilog-cplex-optimization-studio).

## Setup

If you want to use IBM Q / IonQ / DWave cloud services, sign up on the respective web page and follow the instructions to get the required _api token_.

## Config

Open [conf/Config.py](samples/Config.py) and setup the problem to solve. Finally, run:

```bash
$ python Main.py
```

The output will look similar to the following:
```bash
Problem(name='Set Coverage', k=2, S=array([list([7]), list([8]), list([1, 2])], dtype=object), W=[100, 1, 1, 1])
Quantum Device: statevector_simulator
Greedy              [0, 2] -> 102 | 0:00:00.000405
BruteForce          [0, 2] -> 102 | 0:00:00.000173
Qiskit-NumpyExact   [0, 2] -> 102 | 0:00:00.053109
Qiskit-CPLEX        [0, 2] -> 102 | 0:00:00.057020
Qiskit-VQE          [0, 2] -> 102 | 0:00:00.745350
Qiskit-QAOA         [0, 2] -> 102 | 0:00:00.748309
```


