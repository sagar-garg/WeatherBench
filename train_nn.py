from src.train import *

if __name__ == '__main__':
    args = load_args()
    args.pop('my_config')
    train(**args)

