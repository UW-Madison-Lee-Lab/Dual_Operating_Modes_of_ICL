import os
import argparse
import setproctitle
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from setting import Circle, Regular4, Regular6, Regular8, Regular12, D ,PriorProcesser
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
# python train.py --name=GPT
parser = argparse.ArgumentParser(description='PyTorch In-context Learning Training Code')
parser.add_argument('--gpu', default='0', type=str, help='which gpus to use')
parser.add_argument('--dataset', default="gaussian", type=str, help='distribution used to generate the dataset')
parser.add_argument('--model_arch', default='gpt', type=str, help='the model architecture used')
parser.add_argument('--print_freq', default=75, type=int, help='print frequency (default: 75)')
parser.add_argument('--random_seed', default=1, type=int, help='the seed used for torch & numpy')
parser.add_argument('--base_dir', default="./", type=str, help='base directory')

parser.add_argument('--num_heads', default=8, type=int, help='number of heads for multi-headed attention (default: 8)')
parser.add_argument('--depth', default=10, type=int, help='depth of the transformer architecture (default: 12)')
parser.add_argument('--embed_dim', default=1024, type=int, help='embedding dimension of the transformer feature extractor (default: 256)')
parser.add_argument('--steps', default=10, type=int, help='total number of training steps we want to run')#5000
parser.add_argument('--lr', default=0.00001, type=float, help='initial model learning rate')
parser.add_argument('--wd', default=0.00001, type=float, help='weight decay hyperparameter (default: 0.00001)')
parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--epochs', default=3, type=int, help='number of total epochs to run')
parser.add_argument('--num_k', default=32, type=int, help='maximum sequence length of the input (default: 11)')

parser.add_argument('--exp_name', default='None', type=str)

parser.add_argument('--a', default=0, type=int, help='1/b**a')
parser.add_argument('--b', default=4, type=int, help='1/b**a')

parser.add_argument('--M', default=0, type=int, help='M')

parser.add_argument('--input_dim', default=2, type=int, help='d=M')

parser.set_defaults(augment=True)


# Specifying gpu usage using cli hyperparameters
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
setproctitle.setproctitle(args.exp_name)

import time
import torch
import random
import numpy as np
from utils import TransformerModel, FlexDataset
from torch.distributions.multivariate_normal import MultivariateNormal
from mydataloader import Prior

# Seeding programmatic sources of randomness
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)

    
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_model(args, prior, model, optimizer, epoch):
    batch_time = AverageMeter()
    batch_loss = AverageMeter()

    # Switch model to train mode
    model.train()
    end = time.time()
    for i in tqdm(range(args.steps)):

        # Loop through batch of inputs
        #for sample_idx in range(len(inputs)):
        #input_ = torch.stack([inputs[sample_idx]]).cuda()
        #target = torch.stack([targets[sample_idx]]).cuda()
        
        input_, target, label = prior.draw_sequences(bs=args.batch_size, k=args.num_k)
        input_ = torch.from_numpy(input_).cuda().float()
        target = torch.from_numpy(target).cuda().float()
        
        # Forward through the gpt model
        output = model(input_, target)
        
        # Calculate the squared error loss
        loss = (target - output).square().mean()
        
        # Backpropagate gradients and update model
        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # Record the loss and elapsed time
        batch_loss.update(loss.data)
        batch_time.update(time.time() - end)
        end = time.time()

        
        # Print training dataset logs
        #end = time.time()
        #if i % args.print_freq == 0:
        #    print('Epoch: {epoch} [{step}/{total_steps}]\t'
        #          'Time {time.val:.3f} ({time.avg:.3f})\t'
        #          'MSE Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
        #              epoch=epoch, step=i, total_steps=args.steps, 
        #              time=batch_time, loss=batch_loss))
    print(batch_loss.avg)
    return model, optimizer


if 1:
    assert args.model_arch == 'gpt'
    k_list = [i+1 for i in range(args.num_k)]
    k_list = [a-1 for a in k_list]
    k_array = np.array(k_list)
    k_list = [r'${}$'.format(a) for a in k_list]
    
    D_load = 0
    B_flag = 1
    B_load = 0
    T_flag = 1
    T_load = 0
    
    KK = 2000
    BS = args.batch_size
    K = KK*BS
    z = 1
    linewidth = 3.0
    
    if args.exp_name == 'Regular4_delta':
        fig_name = 'regular4_base='+str(args.b)+'_lr='+str(args.lr)+'_Depth='+str(args.depth)+'_EmbDim='+str(args.embed_dim)+'_BS='+str(args.batch_size)+'_Step='+str(args.steps)
    elif args.exp_name == 'RegularM_M':
        fig_name = 'regularM'                  +'_lr='+str(args.lr)+'_Depth='+str(args.depth)+'_EmbDim='+str(args.embed_dim)+'_BS='+str(args.batch_size)+'_Step='+str(args.steps)
    elif args.exp_name == 'D_d':
        fig_name = 'D_d'                       +'_lr='+str(args.lr)+'_Depth='+str(args.depth)+'_EmbDim='+str(args.embed_dim)+'_BS='+str(args.batch_size)+'_Step='+str(args.steps)
    
    hypers_list = []
    if args.exp_name == 'Regular4_delta':
        delta_m_list = [1/args.b**args.a]
        delta_w_list = [1/args.b**args.a]
        for delta_m, delta_w in zip(delta_m_list, delta_w_list):
            hypers_list.append({'delta_m': delta_m,
                                'delta_w': delta_w})
    elif args.exp_name == 'D_d':
        d_list = [args.input_dim]
        delta_m = 1/16 #1/10
        delta_w = 1/16 #1/10
        for d in d_list:
            hypers_list.append({'d': d})
    for hypers in hypers_list:
        if args.exp_name == 'Regular4_delta':
            delta_m = hypers['delta_m']
            delta_w = hypers['delta_w']
            print('***** delta_m=' +str(delta_m)+ ' / delta_w=' +str(delta_w)+ ' *****')
            folder = 'data/Transformer/'+fig_name+'/delta_m='+str(delta_m)+' delta_w='+str(delta_w)+'/'
            prior = Regular4(delta_m**0.5, delta_m**0.5)
            priorProcesser = PriorProcesser(prior)
        if args.exp_name == 'D_d':
            d = hypers['d']
            print('***** d =' +str(d)+ ' *****')
            folder = 'data/Transformer/'+fig_name+'/d='+str(d)+'/'
            
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        if 1: # data
            if not D_load:
                bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(K, args.num_k)
                np.save(folder+'bs_xs.npy', bs_xs)
                np.save(folder+'bs_ys.npy', bs_ys)
                np.save(folder+'bs_retrieval.npy', bs_retrieval)
                np.save(folder+'bs_learning.npy', bs_learning)
            else:
                bs_xs = np.load(folder+'bs_xs.npy')
                bs_ys = np.load(folder+'bs_ys.npy')
                bs_retrieval = np.load(folder+'bs_retrieval.npy')
                bs_learning = np.load(folder+'bs_learning.npy')
        
        if B_flag:
            if not B_load:
                B_preds = np.zeros([K, args.num_k])
                for k in tqdm(range(1, args.num_k+1,1)):
                    for j in range(K):
                        xs, ys = bs_xs[j][:k], bs_ys[j][:k]
                        returned_dict = priorProcesser.predict(xs, ys, priorProcesser.topic_ws[0])
                        B_preds[j, k-1] = returned_dict['prediction']
                
                np.save(folder+'B_preds.npy', B_preds)
            else:
                B_preds = np.load(folder+'B_preds.npy')
        
        if T_flag:
            # Loading in gpt model for training
            model = TransformerModel(n_dims=priorProcesser.d, n_positions=args.num_k, 
                                     n_embd=args.embed_dim, n_layer=args.depth, 
                                     n_head=args.num_heads)
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        # Iterating through training epochs
        for epoch in range(1, args.epochs+1):
            print('******** EP = ' +str(epoch)+ ' *******')
            if not os.path.exists(folder+'EP=' +str(epoch)):
                os.makedirs(folder+'EP=' +str(epoch))
            state_folder = folder+'EP=' +str(epoch)+'/checkpoint_'+str(epoch)+'.pth'
            
            if 1: # Train gpt model for one epoch on generated dataset
                if T_flag and (not T_load):
                    model, optimizer = train_model(args, priorProcesser, model, optimizer, epoch=epoch)
                    state = {"model_state_dict": model.state_dict(), 
                             "optimizer_state_dict": optimizer.state_dict(),
                             "train_epoch": epoch}
                    torch.save(state, state_folder)
            
            if 1: #evaluation
                #state = torch.load(state_folder)
                #model.load_state_dict(state['model_state_dict'])
                #optimizer.load_state_dict(state['optimizer_state_dict']) 
                if T_flag:
                    if not T_load:
                        model.eval()
                        T_preds = []
                        with torch.no_grad():
                            for kk in tqdm(range(KK)):
                                T_preds.append( model(torch.from_numpy(bs_xs[kk*BS:(kk+1)*BS]).cuda().float(), torch.from_numpy(bs_ys[kk*BS:(kk+1)*BS]).cuda().float()).cpu().numpy() )
                        T_preds = np.concatenate(T_preds, axis=0)
                        np.save(folder+'EP=' +str(epoch)+ '/T_preds.npy', T_preds)
                    else:
                        T_preds = np.load(folder+'EP=' +str(epoch)+ '/T_preds.npy')