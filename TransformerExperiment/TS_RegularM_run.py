import numpy as np
import os
gpuIdxStr = ' -gpuIdx 0'
cl_list = [
    'python TS_RegularM.py --M 20 --lr 0.00001 --depth 10 --embed_dim 1024 --batch_size 256 --steps 10000', #
    'python TS_RegularM.py --M 12 --lr 0.00001 --depth 10 --embed_dim 1024 --batch_size 256 --steps 10000', #
    'python TS_RegularM.py --M 8  --lr 0.00001 --depth 10 --embed_dim 1024 --batch_size 256 --steps 10000', #
    'python TS_RegularM.py --M 6  --lr 0.00001 --depth 10 --embed_dim 1024 --batch_size 256 --steps 10000', #
    'python TS_RegularM.py --M 4  --lr 0.00001 --depth 10 --embed_dim 1024 --batch_size 256 --steps 10000', #
    ]
for cl in cl_list:
    os.system(cl)
