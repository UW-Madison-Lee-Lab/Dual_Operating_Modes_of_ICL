import os
import argparse
import setproctitle
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from setting import Circle, Regular4, Regular6, Regular8, Regular12, D ,PriorProcesser

setproctitle.setproctitle('BayesianSimulation')

matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

import time
import torch
import random
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal



# Define a function to generate grid positions for the subplots
def grid_position(row, col, total_cols=4):
    return total_cols * row + col + 1


if 1: # define parameters
    S = 85
    num_k = 256+1
    
    k_list = [0,1,2,4,8,16,32,64,128,256]
    k_str_list = [r'$0$', r'$2^0$', r'$2^1$', r'$2^2$', r'$2^3$', r'$2^4$', r'$2^5$', r'$2^6$', r'$2^7$', r'$2^8$']
    #k_list = [a-1 for a in k_list]
    k_array = np.array(k_list)
    #k_str_list = [r'${}$'.format(a) for a in k_list]
    K = 80000
    z = 1
    linewidth = 10.0
    
    pi_min = 0
    pi_max = 1
    pi_y_ticks = [0, 0.5, 1.0]
    pi_y_ticklabels = [r'$0.0$', r'$0.5$', r'$1.0$']
    dis_min = 0
    dis_max = 2
    err_min = 0
    err_max = 1.0
    
    legend_dis = 1.0
    
    
    base = 9
    #sm_list = [(1/base)**1.5, 1/base, (1/base)**0.5, 1, (base)**0.5, base, (base)**1.5]
    sm_list = [1/base, (1/base)**0.5, 1]#, (base)**0.5, base]
    sm_strs = [#f'${base}^{{-1.5}}$',
               f'$1/81$',
               f'$1/9$',
               f'$1$',
               #f'$9$',
               #f'$81$',
               #f'${base}^{{+1.5}}$',
               ]
    #sw_list = [(1/base)**1.5, 1/base, (1/base)**0.5, 1, (base)**0.5, base, (base)**1.5]
    sw_list = [1/base, (1/base)**0.5, 1]#, (base)**0.5, base]
    sw_strs = [#f'${base}^{{-1.5}}$',
               f'$1/81$',
               f'$1/9$',
               f'$1$',
               #f'$9$',
               #f'$81$',
               #f'${base}^{{+1.5}}$',
               ]
    fig_name = 'Compare'
    #colors = ['#CD1818', '#0E21A0', '#116D6E', '#331D2C']
    M=4
    
    plt.rcParams.update({'font.size': 64})
    
    colors = ['#CD1818', '#116D6E', '#0E21A0']#, '#c924aa']
    ratio_list = [4/4, 5/3, 6/2]#, 7/1] 
    labels = ['Farthest', 'Medium', 'Closest']
    
if 0: # preprocess
    for index1, (sm, sw) in enumerate(zip(sm_list,sw_list)):
        for index2, ratio in enumerate(ratio_list):
            
            #folder = 'data/'+fig_name+'/sm='+str(sm)+' sw='+str(sw)+'/'
            folder = 'data/'+fig_name+'/'+sm_strs[index1].replace('/','-')+'/ratio='+str(ratio)+'/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            print('\n***** ' +str(index1)+ '-' +str(index2)+ ' *****')
            prior = Regular4(sm=sm,sw=sw,ratio=ratio)
            print('\n prior.delta_m = ', prior.delta_m)
            print('\n prior.delta_w = ', prior.delta_w)
            priorProcesser = PriorProcesser(prior)
            #gs_small = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=gs_main[row, col])
            
            #priorProcesser.visualize( fig, gs_small )
            if 1:
                #epoch=0
                print('***************')
                colors = ['#CD1818', '#0E21A0', '#116D6E', '#331D2C']
                if 1:
                    bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(K, num_k)
                    B_preds = np.zeros([K, num_k])
                    B_tpis = np.zeros([K, num_k, prior.M])
                    B_ttopic_ms = np.zeros([K, num_k, prior.M, prior.d])
                    B_ttopic_ws = np.zeros([K, num_k, prior.M, prior.d])
                    B_adv_m = np.zeros([K, num_k, prior.M])
                    B_adv_w = np.zeros([K, num_k, prior.M])
                    B_tw = np.zeros([K, num_k, prior.d])
                    for k in tqdm(k_list):
                        for j in range(K):
                            xs, ys = bs_xs[j][:k+1], bs_ys[j][:k+1]
                            returned_dict = priorProcesser.predict(xs, ys, priorProcesser.topic_ws[0])
                            B_preds    [j, k] = returned_dict['prediction']
                            B_tpis     [j, k] = returned_dict['tpis']
                            B_ttopic_ms[j, k] = returned_dict['ttopic_ms']
                            B_ttopic_ws[j, k] = returned_dict['ttopic_ws']
                            B_adv_m    [j, k] = returned_dict['adv_m']
                            B_adv_w    [j, k] = returned_dict['adv_w']
                            B_tw       [j, k] = returned_dict['tw']
                    #np.save(folder+'bs_xs.npy', bs_xs)
                    #np.save(folder+'bs_ys.npy', bs_ys)
                    #np.save(folder+'bs_retrieval.npy', bs_retrieval)
                    np.save(folder+'bs_learning.npy', bs_learning)
                    np.save(folder+'B_preds.npy', B_preds)
                    #np.save(folder+'B_tpis.npy', B_tpis)
                    #np.save(folder+'B_ttopic_ms.npy', B_ttopic_ms)
                    #np.save(folder+'B_ttopic_ws.npy', B_ttopic_ws)
                    #np.save(folder+'B_adv_m.npy', B_adv_m)
                    #np.save(folder+'B_adv_w.npy', B_adv_w)
                    #np.save(folder+'B_tw.npy', B_tw)
                else:
                    #bs_xs = np.load(folder+'bs_xs.npy')
                    #bs_ys = np.load(folder+'bs_ys.npy')
                    #bs_retrieval = np.load(folder+'bs_retrieval.npy')
                    bs_learning = np.load(folder+'bs_learning.npy')
                    B_preds = np.load(folder+'B_preds.npy')
                    #B_tpis = np.load(folder+'B_tpis.npy')
                    #B_ttopic_ms = np.load(folder+'B_ttopic_ms.npy')
                    #B_ttopic_ws = np.load(folder+'B_ttopic_ws.npy')
                    #B_adv_m = np.load(folder+'B_adv_m.npy')
                    #B_adv_w = np.load(folder+'B_adv_w.npy')
                    #B_tw = np.load(folder+'B_tw.npy')


if 1: # set 1x4, draw sphere
    resize = 10
    column = 4
    fig = plt.figure(figsize=(column*1.25*resize, 1.5*resize))
    # Create a 3x4 grid using GridSpec
    gs_main = gridspec.GridSpec(1, column, figure=fig, width_ratios=[1,1,1,1])
    
    gs_title = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[0, 1:4])
    ax_title = fig.add_subplot(gs_title[0])
    ax_title.axis('off')
    ax_title.set_title('Number of In-Context Examples '+r'$(k)$'+' - Risk '+r'$E[(\mathcal{F}^* - y_{k+1}^{*})^2]$', pad=105, fontsize = S)
         
if 1: # draw three loss trends for closest. middle, furthest
    row_index = 0
    
    ax = fig.add_subplot(gs_main[row_index, 0], projection='3d')
    prior = Regular4(sm=sm_list[0],sw=sw_list[0],ratio=ratio_list,colors=colors)
    prior.visualize(ax, caption='Prior '+r'$\boldsymbol{\mu}_\beta$'+' and In-Context '+r'$\boldsymbol{\mu}^*$')
    
    for index1, (sm, sw) in enumerate(zip(sm_list,sw_list)):
        ax = fig.add_subplot(gs_main[row_index, index1+1])
        for index2, ratio in enumerate(ratio_list):
            folder = 'data/'+fig_name+'/'+sm_strs[index1].replace('/','-')+'/ratio='+str(ratio)+'/'
            print('\n***** ' +str(index1)+ '-' +str(0)+ ' *****')
            #bs_xs = np.load(folder+'bs_xs.npy')
            #bs_ys = np.load(folder+'bs_ys.npy')
            #bs_retrieval = np.load(folder+'bs_retrieval.npy')
            bs_learning = np.load(folder+'bs_learning.npy')
            B_preds = np.load(folder+'B_preds.npy')
            #B_tpis = np.load(folder+'B_tpis.npy')
            #B_ttopic_ms = np.load(folder+'B_ttopic_ms.npy')
            #B_ttopic_ws = np.load(folder+'B_ttopic_ws.npy')
            #B_adv_m = np.load(folder+'B_adv_m.npy')
            #B_adv_w = np.load(folder+'B_adv_w.npy')
            
            prior = Regular4(sm=sm,sw=sw)
            
            #ax = fig.add_subplot(gs_main[row_index, row])
            
            
            TV = (B_preds[:,k_array] - bs_learning[:,k_array])**2
            means = np.mean(TV, axis=0)
            std_devs = np.std(TV, axis=0)
            margin_error = z * std_devs
            lower_bound = means - margin_error
            upper_bound = means + margin_error
                
            ax.plot(means, label=labels[index2], linewidth=linewidth, color=colors[index2])
            #ax.fill_between(range(len(means)), lower_bound, upper_bound, color=colors[index2], alpha=0.15)
            
            '''
            TV = (B_preds[:,k_array] - bs_retrieval[:,k_array])**2
            means = np.mean(TV, axis=0)
            std_devs = np.std(TV, axis=0)
            margin_error = z * std_devs
            lower_bound = means - margin_error
            upper_bound = means + margin_error
            
            ax.plot(means, label=r'$(\mathcal{F}^* - y_{k+1}^{\text{R}})^2$', linewidth=linewidth, color='red')
            ax.fill_between(range(len(means)), lower_bound, upper_bound, color='red', alpha=0.15)
            '''
            locations = list(np.arange(len(k_list)))
            ax.set_xticks([locations[0]]+locations[1:][::2])
            ax.set_xticklabels([k_str_list[0]]+k_str_list[1:][::2], fontsize = S)
            
            ax.yaxis.set_tick_params(labelsize=S)
            ax.set_ylim([-err_min,err_max])
            
            if index2 == 0:
                ax.set_title(r'$\delta_{\mu}=\delta_{w}=$'+' '+sw_strs[index1])
                        
            if index1 == 0:
                ax.set_ylabel('')
                
            if index1 == len(sm_strs)-1:
                ax.legend(loc='right', bbox_to_anchor=(legend_dis, 0.65), fontsize = S*0.8)
            

if 1:
    fig.text(0.625, -0.00, 'Number of In-Context Examples '+r'$(k)$', ha='center', va='center', fontsize=S)
    plt.tight_layout()
    plt.savefig('Figure5.pdf',bbox_inches='tight')
