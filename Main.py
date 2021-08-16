import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from conf.Config import Config

if __name__ == '__main__':
    print(Config.p)
    [print(a.solve(Config.p)) for a in Config.algos]
