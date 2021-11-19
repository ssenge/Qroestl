import csv
import time
import warnings
from datetime import datetime
import numpy as np

from run import TaskConfig

warnings.filterwarnings('ignore', category=DeprecationWarning)


def output(msg):
    print(msg)
    fp.write(f'# {msg}\n')

if __name__ == '__main__':
    with np.printoptions(threshold=20, linewidth=np.inf, precision=3):
        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        with open(f'results/{now}.csv', "wt") as fp:
            writer = csv.writer(fp)
            fp.write("# Optimizer name, objective value, wall clock, opt clock, solution\n")
            output(f'Start at {datetime.now()}')
            for t in TaskConfig.tasks:
                p, approaches = t
                output(f'New problem at {datetime.now()}')
                output(p)
                for a in approaches:
                    output(f'New approach {a} at {datetime.now()}')
                    for optimizer in TaskConfig.optimizers:
                        sol = optimizer.optimize(p, a)
                        writer.writerow(sol.to_list())
                        print(str(sol))
                    output(f'End approach {a} at {datetime.now()}')
                output(f'End problem at {datetime.now()}')
            output(f'End at {datetime.now()}')


