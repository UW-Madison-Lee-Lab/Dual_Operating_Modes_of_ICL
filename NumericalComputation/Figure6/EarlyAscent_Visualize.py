import numpy as np
import matplotlib.pyplot as plt
#import matplotlib 
from scipy.spatial import distance
from tqdm import tqdm
import scipy
import pickle
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os
#from mpl_toolkits import mplot3d
#import matplotlib.cm as cm
#from matplotlib.colors import LinearSegmentedColormap
#matplotlib.rc('text', usetex=True)
#matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
def generate_M_uniform_unit_vectors(M, dimension):
    vecs = np.random.normal(0, 1, (M, dimension))  # Generate k vectors of dimension d
    norms = np.linalg.norm(vecs, axis=1)[:, np.newaxis]  # Compute norms for each vector
    unit_vecs = vecs / norms  # Normalize each vector
    return unit_vecs
def pairwise_distances_scipy(vecs):
    dists = distance.pdist(vecs, 'euclidean')
    #dist_matrix = distance.squareform(dists)
    #return dist_matrix
    return dists
def generate_center(M, dimension, min_distance):
    pbar = tqdm(total=99999999)
    iteration = 0
    while True:
        iteration += 1
        pbar.set_description('*** '+str(iteration)+' ***')
        centers = generate_M_uniform_unit_vectors(M, dimension)
        dists = pairwise_distances_scipy(centers)
        if np.min(dists) > min_distance:
            pbar.close()
            return centers
def generate_mixure_coefficients(M, q):
    if q==1:
        random_points = np.ones(M)
    else:
        random_points = np.random.rand(M) 
        random_points = (random_points - np.min(random_points))/(np.max(random_points)-np.min(random_points))
    return random_points/np.sum(random_points)
class SampleGenerator:
    def __init__(self, d, m_center, w_center, s, t):
        self.d = d
        self.m_center = m_center.copy()
        self.w_center = w_center.copy()
        self.s = s
        self.t = t
    def generate_xy_samples(self, k):
        #print('*****')
        #print(self.m_center)
        #print(self.s*np.eye(self.d))
        #print(k)
        xs = np.random.multivariate_normal(mean=self.m_center, cov=self.s*np.eye(self.d), size=k)
        ys = xs@self.w_center 
        ys = ys + np.random.normal(0, self.t, ys.shape)
        return xs, ys[:,np.newaxis]

def sorted_eigenvalues(matrix):
    eigenvalues = np.linalg.eigh(matrix)[0]
    return sorted(eigenvalues, reverse=True)

class Prior:
    def __init__(self, d, M, m_centers, w_centers, pis, s, t, sm, sw):
        self.d = d
        self.M = M
        self.topic_ms = m_centers.copy()
        self.topic_ws = w_centers.copy()
        print(self.topic_ms)
        print(self.topic_ws)
        self.pis = pis
        self.s = s
        self.t = t
        self.sm = sm
        self.sw = sw
        self.delta_m = (self.sm/self.s)**2
        self.delta_w = (self.sw/self.t)**2
    def calculate_loss(self, xs, ys, target_w=None, icl_w=None):
        k = xs.shape[0] - 1
        I = np.eye(self.d)
        #tpis = numpy_to_decimal(self.pis.copy())
        tpis = self.pis.copy()
        ttopic_ms = self.topic_ms.copy()
        ttopic_ws = self.topic_ws.copy()
        
        D_m = self.delta_m*np.sum(xs,axis=0)
        Dm_I = self.delta_m*(k+1)*I
        IDmI_inv = np.linalg.inv(I+Dm_I)
        
        D_w = self.delta_w*np.sum(xs[:-1]*ys[:-1],axis=0)
        Dw_I = self.delta_w*np.sum(xs[:-1,:,np.newaxis]*xs[:-1,np.newaxis,:],axis=0)
        IDwI_inv = np.linalg.inv(I+Dw_I)
        
        eigenvalues = sorted_eigenvalues(I+Dw_I)
        smallest_eigenvalue = np.min(eigenvalues)
        
        #eigenvalues = sorted_eigenvalues(I+Dw_I)
        #print(ys.shape,eigenvalues)
        adv_m = np.zeros([self.M])
        adv_w = np.zeros([self.M])
        learning_upper = 0
        for b in np.arange(self.M):
            #print('\n')
            #print(self.topic_ms.shape)
            #print(self.topic_ms[b].shape)
            #print(D_m.shape)
            #print((self.topic_ms[b]+D_m).shape)
            #print(IDmI_inv.shape)
            #print((self.topic_ms[b]+D_m).shape)
            #print((self.topic_ms[b]+D_m) @ IDmI_inv @ ((self.topic_ms[b]+D_m).T))
            numer = self.topic_ms[b]@self.topic_ms[b] - \
                (self.topic_ms[b]+D_m) @ IDmI_inv @ ((self.topic_ms[b]+D_m).T)
            denom = 2*(self.sm**2)
            adv_m[b] = numer/denom
            #numer = Decimal(numer)
            #denom = Decimal(denom)
            #c_m = np.exp(-numer/denom)
            
            numer = self.topic_ws[b]@self.topic_ws[b] - \
                (self.topic_ws[b]+D_w) @ IDwI_inv @ ((self.topic_ws[b]+D_w).T)
            denom = 2*(self.sw**2)
            adv_w[b] = numer/denom
            #numer = Decimal(numer)
            #denom = Decimal(denom)
            #c_w = np.exp(-numer/denom)
            
            #print(tpis[b].shape)
            #print(c_m.shape)
            #print(c_w.shape)
            #tpis[b] = tpis[b]*c_m*c_w
            G = IDmI_inv@(self.topic_ms[b]+D_m)
            ttopic_ms[b,:] = G
            G = IDwI_inv@(self.topic_ws[b]+D_w)
            ttopic_ws[b,:] = G
        
        adv_m = adv_m - adv_m[0]
        adv_w = adv_w - adv_w[0]
        #print(adv_m)
        #print(adv_w)
        
        adv = adv_m+adv_w
        tpis = tpis*scipy.special.softmax(-adv)
        
        tpis = tpis/np.sum(tpis)
        prediction = tpis@(ttopic_ws@xs[-1])
        
        tw = tpis@ttopic_ws
        
        #print(tpis)
        #print(ttopic_ws)
        
        l_loss = (prediction-ys[-1])**2
        
        #print((ttopic_ws-target_w).shape)
        #print(ttopic_ws-target_w)
        #print(np.linalg.norm(ttopic_ws-target_w,axis=1).shape)
        #print(tpis.shape)
        #print((np.linalg.norm(ttopic_ws-target_w,axis=1)**2 * tpis).shape)
        learning_upper = np.linalg.norm(xs[-1])**2 * np.sum(np.linalg.norm(self.topic_ws-icl_w,axis=1)**2 * tpis) / (smallest_eigenvalue**2)

        if target_w is None:
            r_loss = None
        else:
            r_loss = (prediction-target_w@xs[-1])**2
            
        return_dict = {
            'l_loss': l_loss,
            'r_loss': r_loss,
            'tw': tw,
            'prediction': prediction,
            'tpis': tpis,
            'ttopic_ms': ttopic_ms,
            'ttopic_ws': ttopic_ws,
            'retrieval': target_w@xs[-1],
            'learning': ys[-1],
            'l_upper': learning_upper,
            'adv_m': adv_m,
            'adv_w': adv_w,
            #'eigenvalues': eigenvalues
            }
        return return_dict
    
def is_closer_to_reference(new_center, reference_center, other_centers):
    new_distance = np.linalg.norm(new_center - reference_center)
    #print('#####################')
    #print(new_distance)
    for other_center in other_centers:
        other_distance = np.linalg.norm(new_center - other_center)
        #print(other_distance)
        if new_distance >= other_distance:
            return False
    return True

def find_new_closest_center(reference_center, other_centers, dimension=3):
    while True:
        new_center = generate_M_uniform_unit_vectors(1, dimension)
        if is_closer_to_reference(new_center, reference_center, other_centers):
            return new_center.reshape([-1])

import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')


demon_list = [0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536]#,131072]

str_list = [r'$0$']
for i in range(0, 18):
    str_list.append(r'$2^{%d}$' % i)
    
    
demon_list = [demon_list[0]]+demon_list[1:]#[::2]
str_list   = [str_list  [0]]+str_list  [1:]#[::2]


#demon_list = [16384,32768]
#demon_list = demon_list[:10]
k_list = [k+1 for k in demon_list]

d_list = [1,2,3,5,8]


save_folder = 'results/'

import pathlib
path = pathlib.Path(save_folder)
path.mkdir(parents=True, exist_ok=True)

load = False
if 1: # run exp
    results = {}
    
    for d in d_list:
        folder1 = str(d)+'/'
        results[d] = {}
        
        # set up prior
        #d = d # dimension
        # number of topic
        q = 1 # different ratio between different pi
        min_distance = 0.05
        
        if d == 1:
            M = 2
            print('generate m_centers')
            m_centers = np.array([[+1],[-1]], dtype=np.float64)
            m_centers = m_centers / np.linalg.norm(m_centers,axis=1).reshape([-1,1])
            print('generate w_centers')
            w_centers = np.array([[-1],[+1]], dtype=np.float64)
            w_centers = w_centers / np.linalg.norm(w_centers,axis=1).reshape([-1,1])
        elif d >= 2:
            # generate centers
            M = 3 
            print('generate m_centers')
            m_centers = np.array([[+1,+1]+[+1]*(d-2), [-1,-1]+[-1]*(d-2), [+1,-1]+[-1]*(d-2)], dtype=np.float64)
            m_centers = m_centers / np.linalg.norm(m_centers,axis=1).reshape([-1,1])
            print('generate w_centers')
            w_centers = np.array([[-1,-1]+[-1]*(d-2), [+1,+1]+[+1]*(d-2), [-1,+1]+[+1]*(d-2)], dtype=np.float64)
            w_centers = w_centers / np.linalg.norm(w_centers,axis=1).reshape([-1,1])
        
        results[d]['M'] = M
        results[d]['m_centers'] = m_centers
        results[d]['w_centers'] = w_centers
        print('generate pis')
        pis = generate_mixure_coefficients(M, q)
    
        variance = 2.0
        s = variance*0.5 # sigma
        t = variance # tau
        sm = 0.05 # sigma_mu
        sw = 0.05 # sigma_w
        
        prior = Prior(d, M, m_centers, w_centers, pis, s, t, sm, sw)
        
        if d == 1:
            print('generate new_m_center')
            new_m_center = np.array([+1])
            print('generate new_w_center')
            new_w_center = np.array([+1])
        elif d >= 2:
            print('generate new_m_center')
            new_m_center = np.array([+1,+1]+[+1]*(d-2))
            new_m_center = new_m_center / np.linalg.norm(new_m_center)
            print('generate new_w_center')
            new_w_center = np.array([+1,+1]+[+1]*(d-2))
            new_w_center = new_w_center / np.linalg.norm(new_w_center)
    
        results[d]['new_m_center'] = new_m_center
        results[d]['new_w_center'] = new_w_center
        
        sampleGenerator = SampleGenerator(d, new_m_center, new_w_center, s=s, t=0)
        retrieval_loss = []
        learning_loss = []
        learning_upper = []
        tpis = []
        tw = []
        print('***************')
        #K=400000
        for i in tqdm(k_list):
            
            folder2 = str(i)+'/'
            path = pathlib.Path(save_folder + folder1 + folder2)
            
            
            learning_loss_update = np.load(save_folder + folder1 + folder2 + 'learning_loss_update.npy')
            learning_upper_update = np.load(save_folder + folder1 + folder2 + 'learning_upper_update.npy')
            tpis_update = np.load(save_folder + folder1 + folder2 + 'tpis_update.npy')
            tw_update = np.load(save_folder + folder1 + folder2 + 'tw_update.npy')
                
            learning_loss.append(learning_loss_update)
            learning_upper.append(learning_upper_update)
            tpis.append(tpis_update)
            tw.append(tw_update)
            #print(return_dict['tw_centers'])
        
        l_loss = np.array(learning_loss)
        l_upper = np.array(learning_upper)
        tpis = np.concatenate(tpis, axis=0)
        tw = np.array(tw)
        
        results[d]['l_loss'] = l_loss
        results[d]['l_upper'] = l_upper
        results[d]['tpis'] = tpis
        results[d]['tw'] = tw

        with open('results.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
if 1: # draw bound
    with open('results.pickle', 'rb') as handle:
        results = pickle.load(handle)
    cut = 20
    fig, axs = plt.subplots(figsize=(30, 14), nrows=2, ncols=5, sharex=True, sharey=False, gridspec_kw={'height_ratios': [2.5, 1]})
    for index, d in enumerate(d_list):
        
        learning_loss = results[d]['l_loss'][:cut]
        learning_upper = results[d]['l_upper'][:cut]
        tpis = results[d]['tpis'][:cut]
        tw = results[d]['tw'][:cut]
        
        #print(cut)
        demonum_list = demon_list[:cut]
        sticks = list(np.arange(len(demonum_list)))
        strs = str_list[:cut]
        
        #print(tpis)
        #cut=100
        # Create the first axis
        ax1 = axs[0,index]
        ax1.set_title(r'$d=$'+' '+str(d), fontsize=40)
        lns1 = ax1.plot(sticks, learning_loss,  linewidth=4, label='Risk of ICL with Correct Labels', color='darkblue')
        #ax1_ = ax1.twinx()
        lns2 = ax1.plot(sticks, learning_upper,  linewidth=4, label='Risk Upper Bound of ICL with Correct Labels', color='cyan', linestyle='dashed')
        #ax1.set_xlabel('number of demonstrations', fontsize=40)
        if index == 0:
            ax1.set_ylabel('Risk/Bound', fontsize=30)#, color='blue')
        #if index == 4:
        #    ax1_.set_ylabel('Upper Bound', fontsize=30)#, color='blue')
        ax1.tick_params(axis='y', labelsize=40)
        ax1.tick_params(axis='x', labelsize=40)
        #ax1_.tick_params(axis='y', labelsize=40)
        #ax1_.tick_params(axis='x', labelsize=40)
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        if index == 2:
            ax1.legend(lns, labs, loc='upper center', fontsize=30, ncol=2, bbox_to_anchor=(0.4, 1.275), edgecolor='black')
        
        
        # Create a twin axis for the ratio
        if d == 1:
            ax2 = axs[1,index]
            ax2.plot(sticks, tpis[:,0], linewidth=4, linestyle='dashed', label='Mixture Weight of Component 1 (Misleading)', color='green')
            ax2.plot(sticks, tpis[:,1], linewidth=4, linestyle='dashed', label='Mixture Weight of Component 2 (Target)', color='#ff4d01')
            
        elif d >= 2:
            ax2 = axs[1,index]
            ax2.plot(sticks, tpis[:,0], linewidth=4, linestyle='dashed', label='Mixture Weight of Component 1 (Misleading)', color='green')
            ax2.plot(sticks, tpis[:,1], linewidth=4, linestyle='dashed', label='Mixture Weight of Component 2 (Target)' , color='#ff4d01')
            ax2.plot(sticks, tpis[:,2], linewidth=4, linestyle='dashed', label='Mixture Weight of Component 3', color='blue')
        if index == 0:
            ax2.set_ylabel('Mixture Weight', fontsize=30)#, color='red')
        ax2.tick_params(axis='y', labelsize=40)#, labelcolor='red')
        ax2.set_xticks( sticks[1:][::4], strs[1:][::4], fontsize=40)
        
        if index == 2:
            ax2.legend(loc='upper center', fontsize=30, ncol=3, bbox_to_anchor=(0.4, 1.55), edgecolor='black')
            
        if d == 3:
            ax2.set_xlabel('Numbser of In-Context Examples '+r'$(k)$', fontsize=30)
            
    #plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    plt.savefig('UpperBound.pdf', bbox_inches='tight')
    
    
if 1: # draw traj
    with open('results.pickle', 'rb') as handle:
        results = pickle.load(handle)
        
    demonum_list = demon_list
    
    fig = plt.figure(figsize=(33,10))
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 0.05, 1, 1])
    for index, d in enumerate([1,2,3]):
        M = results[d]['M']
        
        m_centers = results[d]['m_centers']
        w_centers = results[d]['w_centers']
        
        new_m_center = results[d]['new_m_center']
        new_w_center = results[d]['new_w_center']
        
        tw = results[d]['tw']
        
        
        if d == 1:
            sticks = np.arange(len(demonum_list))
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(sticks, tw, marker='o', markersize=15, linewidth=4, label='Traj of '+r'$E[\widetilde{{\boldsymbol{{w}}}}_{{{}}}]$'+' with increasing '+r'$k$', color='blue')
            ax.set_xticks(sticks[::2], demonum_list[::2], fontsize=40) 
            ax.axhline(y = -1.0, color = 'green', linestyle = 'dashed', linewidth=4, label=r'$\boldsymbol{{w}}_{{{1}}}$' + ' of Center 1 (Misleading)')
            ax.axhline(y = +1.0, color = '#ff4d01', linestyle = 'dashed', linewidth=4, label=r'$\boldsymbol{{w}}_{{{2}}}$' + ' of Center 2 (Target)')
            ax.tick_params(axis='y', labelsize=40)
            ax.tick_params(axis='x', labelsize=40)
            
            ax.set_xticks( sticks[1:][::4], strs[1:][::4], fontsize=40)
            
            ax.set_xlabel('Number of In-Context Examples '+r'$(k)$', fontsize = 40)
            ax.set_ylabel('Value of '+r'$w$', fontsize = 40)
            ax.legend(fontsize = 30, loc='upper center', bbox_to_anchor=(0.5, 0.95), edgecolor='black')
        
        if d == 2:
            S = 300
            ax = fig.add_subplot(gs[0, 2])
            ax.scatter(w_centers[0][0], w_centers[0][1], s=S, c = 'green'  , label=r'$\boldsymbol{{w}}_{{{1}}}$' + ' of Center 1 (Misleading)')
            ax.scatter(w_centers[1][0], w_centers[1][1], s=S, c = '#ff4d01', label=r'$\boldsymbol{{w}}_{{{2}}}$' + ' of Center 2 (Target)')
            ax.scatter(w_centers[2][0], w_centers[2][1], s=S, c = 'gray'   , label=r'$\boldsymbol{{w}}_{{{3}}}$' + ' of Center 3')
            points = np.stack(tw, axis=0)
            ax.plot(points[:, 0], points[:, 1], marker='o', markersize=15, linestyle='-', linewidth=4, color='blue', label='Traj of '+r'$E[\widetilde{{\boldsymbol{{w}}}}_{{{}}}]$'+' with increasing '+r'$k$')

            # Add text labels to each point
            for index, i in enumerate(range(len(points))):
                if k_list[i]-1 in [0]:
                    ax.text(points[i, 0], points[i, 1], r'$k=0$', fontsize=40, verticalalignment='bottom', horizontalalignment='right')
                if k_list[i]-1 in [32, 1024]:
                    mapp = {
                        32: r'$k=2^5$',
                        1024: r'$k=2^{10}$',
                        32768: r'$k=2^{15}$',
                        }
                    ax.text(points[i, 0], points[i, 1], mapp[k_list[i]-1], fontsize=40, verticalalignment='top', horizontalalignment='left')
                if k_list[i]-1 in [32768]:
                    mapp = {
                        32: r'$k=2^5$',
                        1024: r'$k=2^{10}$',
                        32768: r'$k=2^{15}$',
                        }
                    ax.text(points[i, 0], points[i, 1], mapp[k_list[i]-1], fontsize=40, verticalalignment='bottom', horizontalalignment='right')
            ax.tick_params(axis='y', labelsize=40)
            ax.tick_params(axis='x', labelsize=40)
            ax.set_xlabel('Value of first dimension of '+r'$\boldsymbol{{w}}$', fontsize = 40)
            ax.set_ylabel('Value of second dimension of '+r'$\boldsymbol{{w}}$', fontsize = 40)
            ax.legend(fontsize = 30, loc='upper center', bbox_to_anchor=(0.5, 0.95), edgecolor='black')
            
        if d in [3]:
            flag = 'w'
            ax = fig.add_subplot(gs[0, 3], projection='3d')
            #ax = plt.axes(projection='3d')
            ax.view_init(75, 75)
            topics = w_centers
            new_task = new_w_center
            resize = 0.35
            # Plotting the unit sphere with higher transparency
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='c', alpha=0.2)
  
            if flag == 'm':
                ax.scatter(topics[0,0], topics[0,1], topics[0,2], s=800*resize, c='green'  , marker='o', depthshade=False, label = r'$\boldsymbol{\mu}_\beta$')
                ax.scatter(topics[1,0], topics[1,1], topics[1,2], s=800*resize, c='#ff4d01', marker='o', depthshade=False, label = r'$\boldsymbol{\mu}_\beta$')
                ax.scatter(topics[2,0], topics[2,1], topics[2,2], s=800*resize, c='gray'   , marker='o', depthshade=False, label = r'$\boldsymbol{\mu}_\beta$')
            if flag == 'w':
                ax.scatter(topics[0,0], topics[0,1], topics[0,2], s=800*resize, c='green'  , marker='o', depthshade=False, label = r'$\boldsymbol{{w}}_{{{1}}}$' + ' of Center 1 (Misleading)')
                ax.scatter(topics[1,0], topics[1,1], topics[1,2], s=800*resize, c='#ff4d01', marker='o', depthshade=False, label = r'$\boldsymbol{{w}}_{{{2}}}$' + ' of Center 2 (Target)')
                ax.scatter(topics[2,0], topics[2,1], topics[2,2], s=800*resize, c='gray'   , marker='o', depthshade=False, label = r'$\boldsymbol{{w}}_{{{3}}}$' + ' of Center 3')
            
            plt.legend()
            
            points = np.stack(tw, axis=0)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], marker='o', markersize=10, color='blue', linestyle='-', linewidth=4, label='Traj of '+r'$E[\widetilde{{\boldsymbol{{w}}}}_{{{}}}]$'+' with increasing '+r'$k$')
            
            # Add text labels to each point
            #for i in range(len(points)):
            for index, i in enumerate(range(len(points))):
                if k_list[i]-1 in [32,1024,32768]:
                    mapp = {
                        32: r'$k=2^5$',
                        1024: r'$k=2^{10}$',
                        32768: r'$k=2^{15}$',
                        }
                    ax.text(points[i, 0], points[i, 1], points[i, 2], mapp[k_list[i]-1], fontsize=30, verticalalignment='bottom', horizontalalignment='right')
                if k_list[i]-1 in [0]:
                    ax.text(points[i, 0], points[i, 1], points[i, 2], r'$k=0$', fontsize=30, verticalalignment='top', horizontalalignment='left')
            
            ax.tick_params(axis='z', labelsize=30)
            ax.tick_params(axis='y', labelsize=30)
            ax.tick_params(axis='x', labelsize=30)
            plt.legend(fontsize = 30, loc='upper center', framealpha=0.75, bbox_to_anchor=(0.5, 0.2), edgecolor='black')
        #plt.subplots_adjust(wspace=(0.3,0))#, hspace=0.4)
        plt.savefig('traj.pdf',bbox_inches='tight')
