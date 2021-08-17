import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from samples.Config import Config

if __name__ == '__main__':
    print(Config.p)
    print(f'Quantum Device: {Config.qdev.backend_name}')
    [print(f'{str(a) : <20}{str(a.solve(Config.p))}') for a in Config.algos]
