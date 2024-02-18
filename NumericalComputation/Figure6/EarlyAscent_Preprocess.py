import numpy as np
import matplotlib.pyplot as plt
#import matplotlib 
from scipy.spatial import distance
from tqdm import tqdm
import scipy
import pickle
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
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


d_list = [1,2,3,5,8]
demon_list = [0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072]
#demon_list = demon_list[:10]
k_list = [k+1 for k in demon_list]
K=10000

save_folder = 'results/'

import pathlib
path = pathlib.Path(save_folder)
path.mkdir(parents=True, exist_ok=True)

load = False
if 1: # run exp
    for d in d_list:
        folder1 = str(d)+'/'
        
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
        
        sampleGenerator = SampleGenerator(d, new_m_center, new_w_center, s=s, t=0)
        retrieval_loss = []
        learning_loss = []
        learning_upper = []
        tpis = []
        tw = []
        print('***************')
        for i in tqdm(k_list):
            
            folder2 = str(i)+'/'
            path = pathlib.Path(save_folder + folder1 + folder2)
            path.mkdir(parents=True, exist_ok=True)
            
            if load == False:
                #print('******************************')
                r_sum = 0
                l_sum = 0
                tpis_sum = 0
                tw_sum = 0
                l_upper_sum = 0
                
                for j in range(K):
                    xs, ys = sampleGenerator.generate_xy_samples(i)
                    #xs,ys = np.ones([1,3]),np.zeros([1,1])
                    return_dict = prior.calculate_loss(xs, ys, w_centers[0], new_w_center)
                    #r_sum += return_dict['r_loss']
                    l_sum += return_dict['l_loss']
                    tpis_sum += return_dict['tpis']
                    tw_sum += return_dict['tw']
                    l_upper_sum += return_dict['l_upper']
                    #print(return_dict['tpis'])
                    
                #retrieval_loss.append(r_sum/K)
                learning_loss_update = l_sum/K
                learning_upper_update = l_upper_sum/K
                tpis_update = tpis_sum.reshape([1,M])/K
                tw_update = tw_sum/K
                
                np.save(save_folder + folder1 + folder2+'learning_loss_update.npy', learning_loss_update)
                np.save(save_folder + folder1 + folder2+'learning_upper_update.npy', learning_upper_update)
                np.save(save_folder + folder1 + folder2+'tpis_update.npy', tpis_update)
                np.save(save_folder + folder1 + folder2+'tw_update.npy', tw_update)
            else:
                learning_loss_update = np.load(save_folder + folder1 + folder2+'learning_loss_update.npy')
                learning_upper_update = np.load(save_folder + folder1 + folder2+'learning_upper_update.npy')
                tpis_update = np.load(save_folder + folder1 + folder2+'tpis_update.npy')
                tw_update = np.load(save_folder + folder1 + folder2+'tw_update.npy')