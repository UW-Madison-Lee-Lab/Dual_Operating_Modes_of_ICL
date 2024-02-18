import numpy as np
import os
gpuIdxStr = ' -gpuIdx 0'
cl_list = [
    'python TS_Regular4_delta.py --exp_name Regular4_delta --a 0 --b 4 --lr 0.00001 --depth 10 --embed_dim 1024 --batch_size 256 --steps 10000', #
    'python TS_Regular4_delta.py --exp_name Regular4_delta --a 1 --b 4 --lr 0.00001 --depth 10 --embed_dim 1024 --batch_size 256 --steps 10000', #
    'python TS_Regular4_delta.py --exp_name Regular4_delta --a 2 --b 4 --lr 0.00001 --depth 10 --embed_dim 1024 --batch_size 256 --steps 10000', #
    'python TS_Regular4_delta.py --exp_name Regular4_delta --a 3 --b 4 --lr 0.00001 --depth 10 --embed_dim 1024 --batch_size 256 --steps 10000', #
    'python TS_Regular4_delta.py --exp_name Regular4_delta --a 4 --b 4 --lr 0.00001 --depth 10 --embed_dim 1024 --batch_size 256 --steps 10000', #
    ]
for cl in cl_list:
    os.system(cl)
