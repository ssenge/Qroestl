# Qroestl
A thin optimization layer on top of Qiskit. Named after the delicious Austrian & Bavarian dish [Gröstl](https://de.wikipedia.org/wiki/Tiroler_Gröstl) which I ate when I started this ;-)

## Install

First clone the repo:
```bash
$ cgit clone https://github.com/ssenge/Qroestl.git
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

Open [conf/Config.py](conf/Config.py) and setup the problem to solve. Finally, run:

```bash
$ python Main.py
```



