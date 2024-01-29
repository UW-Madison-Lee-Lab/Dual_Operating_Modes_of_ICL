import numpy as np
import os
gpuIdxStr = '0'
cl_list = [
    'python TS_D_d.py --input_dim 2  --lr 0.00001 --depth 10 --embed_dim 1024 --batch_size 256 --steps 10000 --gpu '+gpuIdxStr,
    'python TS_D_d.py --input_dim 4  --lr 0.00001 --depth 10 --embed_dim 1024 --batch_size 256 --steps 10000 --gpu '+gpuIdxStr,
    'python TS_D_d.py --input_dim 8  --lr 0.00001 --depth 10 --embed_dim 1024 --batch_size 256 --steps 10000 --gpu '+gpuIdxStr,
    'python TS_D_d.py --input_dim 16 --lr 0.00001 --depth 10 --embed_dim 1024 --batch_size 256 --steps 10000 --gpu '+gpuIdxStr,
    'python TS_D_d.py --input_dim 32 --lr 0.00001 --depth 10 --embed_dim 1024 --batch_size 256 --steps 10000 --gpu '+gpuIdxStr,
    ]
for cl in cl_list:
    os.system(cl)
