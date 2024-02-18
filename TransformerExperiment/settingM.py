import os
import argparse
import setproctitle
from tqdm import tqdm
#from multiprocessing import Pool
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import numpy as np
import scipy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from decimal import Decimal, getcontext


def numpy_to_decimal(arr, precision=2000):
    """
    Convert a numpy array to an array of decimal.Decimal objects.
    
    Parameters:
    - arr: The numpy array to convert.
    - precision: The desired precision for the Decimal objects.
    
    Returns:
    - A numpy array of decimal.Decimal objects.
    """
    # Set the precision for decimal
    getcontext().prec = precision
    
    # Vectorized function to apply Decimal conversion element-wise
    vec_decimal = np.vectorize(Decimal)
    
    return vec_decimal(arr)

def decimal_to_numpy(decimal_arr, dtype=float):
    """
    Convert a numpy array of decimal.Decimal objects to a standard numpy array.
    
    Parameters:
    - decimal_arr: The numpy array of decimal.Decimal objects to convert.
    - dtype: The desired data type for the returned numpy array (default is float).
    
    Returns:
    - A numpy array with the specified data type.
    """
    # Vectorized function to convert Decimal objects to the desired data type
    vec_float = np.vectorize(lambda x: dtype(str(x)))
    
    return vec_float(decimal_arr)

# Set the precision to 50 decimal places
getcontext().prec = 50


# python train.py --name=GPT
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
class SampleGenerator:
    def __init__(self, d, m_center, w_center, s):
        self.d = d
        self.m_center = m_center.copy()
        self.w_center = w_center.copy()
        self.s = s
    def generate_xy_samples(self, k):
        xs = np.random.multivariate_normal(mean=self.m_center, cov=self.s*np.eye(self.d), size=k)
        ys = xs@self.w_center
        return xs, ys[:,np.newaxis]

class D:
    def __init__(self, D_num = 4, sm=0.25, sw=0.25, rotate = 0):
        self.d = D_num
        self.M = D_num
        
        self.topic_ms = np.zeros([self.M, D_num])
        for d in range(D_num):
            self.topic_ms[d,d] = 1
            #self.topic_ms[D_num+d,d] = -1
        self.topic_ws = self.topic_ms.copy()
        
        self.pis = np.ones([self.M])/self.M
        self.s = 1
        self.x_covariance = self.s**2 * np.eye(self.d)
        self.t = 1
        self.y_variance = self.t**2
        self.sm = sm
        self.m_covariance = self.sm**2 * np.eye(self.d)
        self.sw = sw
        self.w_covariance = self.sw**2 * np.eye(self.d)
        self.delta_m = (self.sm/self.s)**2
        self.delta_w = (self.sw/self.t)**2
        
        new_task_m = 3*self.topic_ms[0] + self.topic_ms[1]
        self.new_task_m = new_task_m / np.linalg.norm(new_task_m)
        new_task_w = 3*self.topic_ws[0] + self.topic_ws[1]
        self.new_task_w = new_task_w / np.linalg.norm(new_task_w)
        
class Circle:
    def __init__(self, topic_num = 4, sm=0.25, sw=0.25, rotate = 0):
        self.d = 2
        self.M = topic_num
        theta_list = np.array([2*np.pi*i/topic_num for i in range(topic_num)])
        self.topic_ms = np.array([[np.cos(theta),np.sin(theta)] for theta in theta_list])
        for i in range(rotate):
            theta_list = np.concatenate([theta_list[1:],theta_list[:1]])
        self.topic_ws = np.array([[np.cos(theta),np.sin(theta)] for theta in theta_list])

        
        self.pis = np.ones([self.M])/self.M
        self.s = 1
        self.x_covariance = self.s**2 * np.eye(self.d)
        self.t = 1
        self.y_variance = self.t**2
        self.sm = sm
        self.m_covariance = self.sm**2 * np.eye(self.d)
        self.sw = sw
        self.w_covariance = self.sw**2 * np.eye(self.d)
        self.delta_m = (self.sm/self.s)**2
        self.delta_w = (self.sw/self.t)**2
        
        new_task_m = 3*self.topic_ms[0] + self.topic_ms[1]
        self.new_task_m = new_task_m / np.linalg.norm(new_task_m)
        new_task_w = 3*self.topic_ws[0] + self.topic_ws[1]
        self.new_task_w = new_task_w / np.linalg.norm(new_task_w)
        
class Old4:
    def __init__(self, sm=0.25, sw=0.25):
        self.d = 3
        self.M = 4
        self.topic_ms = np.array([[+1, 0,+1],[-1, 0,+1],[ 0,+1,-1],[ 0,-1,-1]],dtype=np.float64)/np.sqrt(2)


        self.topic_ws = np.array([[+1, 0,+1],[-1, 0,+1],[ 0,+1,-1],[ 0,-1,-1]],dtype=np.float64)/np.sqrt(2)
        
        self.pis = np.ones([self.M])/self.M
        self.s = 1
        self.x_covariance = self.s**2 * np.eye(self.d)
        self.t = 1
        self.y_variance = self.t**2
        self.sm = sm
        self.m_covariance = self.sm**2 * np.eye(self.d)
        self.sw = sw
        self.w_covariance = self.sw**2 * np.eye(self.d)
        self.delta_m = (self.sm/self.s)**2
        self.delta_w = (self.sw/self.t)**2
        
        new_task_m = 3*self.topic_ms[0] - self.topic_ms[1]
        self.new_task_m = new_task_m / np.linalg.norm(new_task_m)
        new_task_w = 3*self.topic_ws[0] - self.topic_ws[1]
        self.new_task_w = new_task_w / np.linalg.norm(new_task_w)

class Regular4:
    def __init__(self, sm=0.25, sw=0.25, match = True):
        self.d = 3
        self.M = 4
        self.topic_ms = np.array([[0,0,-1],
                                  [(8/9)**0.5, 0, 1/3],
                                  [-(2/9)**0.5, +(2/3)**0.5, 1/3],
                                  [-(2/9)**0.5, -(2/3)**0.5, 1/3]])
        self.topic_ms = self.topic_ms[:,[0,2,1]]
        self.topic_ws = self.topic_ms.copy()
        if match == False:
            self.topic_ws = self.topic_ws[[1,2,3,0]]
        
        self.pis = np.ones([self.M])/self.M
        self.s = 1
        self.x_covariance = self.s**2 * np.eye(self.d)
        self.t = 1
        self.y_variance = self.t**2
        self.sm = sm
        self.m_covariance = self.sm**2 * np.eye(self.d)
        self.sw = sw
        self.w_covariance = self.sw**2 * np.eye(self.d)
        self.delta_m = (self.sm/self.s)**2
        self.delta_w = (self.sw/self.t)**2
        
        new_task_m = 2*self.topic_ms[0] + self.topic_ms[1]
        self.new_task_m = new_task_m / np.linalg.norm(new_task_m)
        new_task_w = 2*self.topic_ws[0] + self.topic_ws[1]
        self.new_task_w = new_task_w / np.linalg.norm(new_task_w)
    
    def visualize(self):
        # Define edge colors
        edge_colors = ['black', 'black', 'gray', 'black', 'black', 'black']
        
        edges_ms = [
            [self.topic_ms[0], self.topic_ms[1]],
            [self.topic_ms[0], self.topic_ms[2]],
            [self.topic_ms[0], self.topic_ms[3]],
            [self.topic_ms[1], self.topic_ms[2]],
            [self.topic_ms[1], self.topic_ms[3]],
            [self.topic_ms[2], self.topic_ms[3]]
        ]
        
        edges_ws = [
            [self.topic_ws[0], self.topic_ws[1]],
            [self.topic_ws[0], self.topic_ws[2]],
            [self.topic_ws[0], self.topic_ws[3]],
            [self.topic_ws[1], self.topic_ws[2]],
            [self.topic_ws[1], self.topic_ws[3]],
            [self.topic_ws[2], self.topic_ws[3]]
        ]
        
        self.resize = 3
        fig = plt.figure(figsize=(9*self.resize, 5*self.resize))
        
        # First subplot for topic_ms
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_tetrahedron('m', ax1, self.topic_ms, edges_ms, self.new_task_m, edge_colors)
        ax1.set_title(r'$\boldsymbol{\mu}_\beta$'+' of Prior and '+r'$\boldsymbol{\mu}^*$'+' of Demonstrations')
        ax1.view_init(60, 70)
        
        ax1.legend()
        
        # Second subplot for topic_ws
        ax2 = fig.add_subplot(122, projection='3d')
        self._plot_tetrahedron('w', ax2, self.topic_ws, edges_ws, self.new_task_w, edge_colors)
        ax2.set_title(r'$\boldsymbol{w}_\beta$'+' of Prior and '+r'$\boldsymbol{w}^*$'+' of Demonstrations')
        ax2.view_init(60, 70)
        
        ax2.legend()
        
        #plt.show()

    def _plot_tetrahedron(self, flag, ax, topics, edges, new_task, edge_colors):
        # Plotting the unit sphere with higher transparency
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='c', alpha=0.2)
        
        # Plot edges with specified colors
        for i, edge in enumerate(edges):
            edge = np.array(edge)
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], c=edge_colors[i], linewidth=2.5)
        
        # Find the closest point to the new_task (green star)
        distances = np.linalg.norm(topics - new_task, axis=1)
        closest_index = np.argmin(distances)
        
        # Plot lines from the origin to the vertices and the additional point
        for topic in topics:
            ax.plot([0, topic[0]], [0, topic[1]], [0, topic[2]], c='purple', linestyle='--', linewidth=1.5)
        ax.plot([0, new_task[0]], [0, new_task[1]], [0, new_task[2]], c='purple', linestyle='--', linewidth=1.5)
        
        # Plot the vertices using large orange circles, label them, and add a red circle to the closest point
        if flag == 'm':
            ax.scatter(topics[:,0], topics[:,1], topics[:,2], s=800*self.resize, c='orange', marker='o', depthshade=False, label = r'$\boldsymbol{\mu}_\beta$')
        if flag == 'w':
            ax.scatter(topics[:,0], topics[:,1], topics[:,2], s=800*self.resize, c='orange', marker='o', depthshade=False, label = r'$\boldsymbol{w}_\beta$')
        
        for i, topic in enumerate(topics):
            ax.text(topic[0], topic[1], topic[2], str(i + 1), fontsize=10*self.resize, ha='center', va='center', zorder=10)
            if i == closest_index:
                if flag == 'm':
                    ax.scatter(topic[0], topic[1], topic[2], s=900*self.resize, facecolors='none', edgecolors='red', linewidth=1.5*self.resize, depthshade=False, label = r'$\boldsymbol{\mu}_\alpha$')
                if flag == 'w':
                    ax.scatter(topic[0], topic[1], topic[2], s=900*self.resize, facecolors='none', edgecolors='red', linewidth=1.5*self.resize, depthshade=False, label = r'$\boldsymbol{w}_\alpha$')
        
        # Plot the additional point using a large green star marker
        if flag == 'm':
            ax.scatter(new_task[0], new_task[1], new_task[2], s=800*self.resize, c='green', marker='*', depthshade=False, label = r'$\boldsymbol{\mu}^*$')
        if flag == 'w':
            ax.scatter(new_task[0], new_task[1], new_task[2], s=800*self.resize, c='green', marker='*', depthshade=False, label = r'$\boldsymbol{w}^*$')
        

class Regular6:
    def __init__(self, sm=0.25, sw=0.25):
        self.d = 3
        self.M = 6
        self.topic_ms = np.array([[+1,0,0],[0,+1,0],[0,0,+1],
                                  [-1,0,0],[0,-1,0],[0,0,-1]],dtype=np.float64)
        self.topic_ws = np.array([[+1,0,0],[0,+1,0],[0,0,+1],
                                  [-1,0,0],[0,-1,0],[0,0,-1]],dtype=np.float64)
        
        self.pis = np.ones([self.M])/self.M
        self.s = 1
        self.x_covariance = self.s**2 * np.eye(self.d)
        self.t = 1
        self.y_variance = self.t**2
        self.sm = sm
        self.m_covariance = self.sm**2 * np.eye(self.d)
        self.sw = sw
        self.w_covariance = self.sw**2 * np.eye(self.d)
        self.delta_m = (self.sm/self.s)**2
        self.delta_w = (self.sw/self.t)**2
        
        new_task_m = 3*self.topic_ms[0] + self.topic_ms[1]
        self.new_task_m = new_task_m / np.linalg.norm(new_task_m)
        new_task_w = 3*self.topic_ws[0] + self.topic_ws[1]
        self.new_task_w = new_task_w / np.linalg.norm(new_task_w)

class Regular8:
    def __init__(self, sm=0.25, sw=0.25):
        self.d = 3
        self.M = 8
        self.topic_ms = np.array([[+1,+1,+1],[+1,+1,-1],
                                  [+1,-1,+1],[+1,-1,-1],
                                  [-1,+1,+1],[-1,+1,-1],
                                  [-1,-1,+1],[-1,-1,-1]],dtype=np.float64)/(3**0.5)
        self.topic_ws = np.array([[+1,+1,+1],[+1,+1,-1],
                                  [+1,-1,+1],[+1,-1,-1],
                                  [-1,+1,+1],[-1,+1,-1],
                                  [-1,-1,+1],[-1,-1,-1]],dtype=np.float64)/(3**0.5)
        
        self.pis = np.ones([self.M])/self.M
        self.s = 1
        self.x_covariance = self.s**2 * np.eye(self.d)
        self.t = 1
        self.y_variance = self.t**2
        self.sm = sm
        self.m_covariance = self.sm**2 * np.eye(self.d)
        self.sw = sw
        self.w_covariance = self.sw**2 * np.eye(self.d)
        self.delta_m = (self.sm/self.s)**2
        self.delta_w = (self.sw/self.t)**2
        
        new_task_m = 3*self.topic_ms[0] + self.topic_ms[1]
        self.new_task_m = new_task_m / np.linalg.norm(new_task_m)
        new_task_w = 3*self.topic_ws[0] + self.topic_ws[1]
        self.new_task_w = new_task_w / np.linalg.norm(new_task_w)

class Regular12:
    def __init__(self, sm=0.25, sw=0.25):
        self.d = 3
        self.M = 12
        self.topic_ms = np.array([[+1,+1,+0],[+1,+0,+1],[+0,+1,+1],
                                  [+1,-1,+0],[+1,+0,-1],[+0,+1,-1],
                                  [-1,+1,+0],[-1,+0,+1],[+0,-1,+1],
                                  [-1,-1,+0],[-1,+0,-1],[+0,-1,-1],],dtype=np.float64)/(2**0.5)
        self.topic_ws = np.array([[+1,+1,+0],[+1,+0,+1],[+0,+1,+1],
                                  [+1,-1,+0],[+1,+0,-1],[+0,+1,-1],
                                  [-1,+1,+0],[-1,+0,+1],[+0,-1,+1],
                                  [-1,-1,+0],[-1,+0,-1],[+0,-1,-1],],dtype=np.float64)/(2**0.5)
        
        self.pis = np.ones([self.M])/self.M
        self.s = 1
        self.x_covariance = self.s**2 * np.eye(self.d)
        self.t = 1
        self.y_variance = self.t**2
        self.sm = sm
        self.m_covariance = self.sm**2 * np.eye(self.d)
        self.sw = sw
        self.w_covariance = self.sw**2 * np.eye(self.d)
        self.delta_m = (self.sm/self.s)**2
        self.delta_w = (self.sw/self.t)**2
        
        new_task_m = 3*self.topic_ms[0] + self.topic_ms[1]
        self.new_task_m = new_task_m / np.linalg.norm(new_task_m)
        new_task_w = 3*self.topic_ws[0] + self.topic_ws[1]
        self.new_task_w = new_task_w / np.linalg.norm(new_task_w)

class Regular20:
    def __init__(self, sm=0.25, sw=0.25):
        self.d = 3
        self.M = 20
        phi = (1+5**0.5)/2
        self.topic_ms = np.array([[+1,+1,+1],
                                  [0,+1/phi,+phi],[+1/phi,+phi,0],[+phi,0,+1/phi],
                                  [0,+1/phi,-phi],[+1/phi,-phi,0],[+phi,0,-1/phi],
                                  [0,-1/phi,+phi],[-1/phi,+phi,0],[-phi,0,+1/phi],
                                  [0,-1/phi,-phi],[-1/phi,-phi,0],[-phi,0,-1/phi],
                                  [+1,+1,-1],
                                  [+1,-1,+1],[+1,-1,-1],
                                  [-1,+1,+1],[-1,+1,-1],
                                  [-1,-1,+1],[-1,-1,-1],],dtype=np.float64)/(3**0.5)
        self.topic_ws = np.array([[+1,+1,+1],[+1,+1,-1],
                                  [0,+1/phi,+phi],[+1/phi,+phi,0],[+phi,0,+1/phi],
                                  [0,+1/phi,-phi],[+1/phi,-phi,0],[+phi,0,-1/phi],
                                  [0,-1/phi,+phi],[-1/phi,+phi,0],[-phi,0,+1/phi],
                                  [0,-1/phi,-phi],[-1/phi,-phi,0],[-phi,0,-1/phi],
                                  [+1,-1,+1],[+1,-1,-1],
                                  [-1,+1,+1],[-1,+1,-1],
                                  [-1,-1,+1],[-1,-1,-1]],dtype=np.float64)/(3**0.5)
        
        self.pis = np.ones([self.M])/self.M
        self.s = 1
        self.x_covariance = self.s**2 * np.eye(self.d)
        self.t = 1
        self.y_variance = self.t**2
        self.sm = sm
        self.m_covariance = self.sm**2 * np.eye(self.d)
        self.sw = sw
        self.w_covariance = self.sw**2 * np.eye(self.d)
        self.delta_m = (self.sm/self.s)**2
        self.delta_w = (self.sw/self.t)**2
        
        new_task_m = 3*self.topic_ms[0] + self.topic_ms[1]
        self.new_task_m = new_task_m / np.linalg.norm(new_task_m)
        new_task_w = 3*self.topic_ws[0] + self.topic_ws[1]
        self.new_task_w = new_task_w / np.linalg.norm(new_task_w)
        
        
class PriorProcesser:
    def __init__(self, prior):
        self.d = prior.d
        self.M = prior.M
        self.topic_ms = prior.topic_ms
        self.topic_ws = prior.topic_ws
        self.pis = prior.pis
        self.s = prior.s
        self.x_covariance = prior.x_covariance
        self.t = prior.t
        self.y_variance = prior.y_variance
        self.sm = prior.sm
        self.m_covariance = prior.m_covariance
        self.sw = prior.sw
        self.w_covariance = prior.w_covariance
        self.delta_m = prior.delta_m
        self.delta_w = prior.delta_w
        
        self.new_task_m = prior.new_task_m
        self.new_task_w = prior.new_task_w
        
        
    def draw_topic(self, ):
        index = np.random.choice(self.M, p=self.pis)
        return self.topic_ms[index], self.topic_ws[index]
    
    def draw_topics(self, num):
        indexes = np.random.choice(self.M, p=self.pis, size=num)
        return self.topic_ms[indexes], self.topic_ws[indexes]
    
    def draw_task(self, topic_m, topic_w):
        task_m = np.random.multivariate_normal(topic_m, self.m_covariance)
        task_w = np.random.multivariate_normal(topic_w, self.w_covariance)
        return task_m, task_w
    
    def draw_tasks(self, topic_m, topic_w, num):
        task_ms = np.random.multivariate_normal(topic_m, self.m_covariance, size=num)
        task_ws = np.random.multivariate_normal(topic_w, self.w_covariance, size=num)
        return task_ms, task_ws
    
    def draw_sequence(self, k):
        topic_m, topic_w = self.draw_topic()
        task_m, task_w = self.draw_task(topic_m, topic_w)
        xs = np.random.multivariate_normal(task_m, self.x_covariance, size=k+1)
        ys = xs@task_w + np.random.normal(0, self.y_variance, size=k+1)
        return xs, ys
    
    def draw_sequences(self, bs, k):
        topic_ms, topic_ws = self.draw_topics(bs)
        bs_xs = []
        bs_ys = []
        bs_zs = []
        for i in range(bs):
            task_m, task_w = self.draw_task(topic_ms[i], topic_ws[i])
            xs = np.random.multivariate_normal(task_m, self.x_covariance, size=k)
            bs_xs.append(xs)
            zs = xs@task_w
            bs_zs.append(zs)
            ys = zs + np.random.normal(0, self.y_variance, size=k)
            bs_ys.append(ys)
        bs_xs = np.stack(bs_xs, axis=0)
        bs_ys = np.stack(bs_ys, axis=0)
        bs_zs = np.stack(bs_zs, axis=0)
        return bs_xs, bs_ys, bs_ys
    #def arxiv_draw_sequences(self, bs, k, num_processes=12):
    #    topic_ms, topic_ws = self.draw_topics(bs)
    #    samples = []
    #    with Pool(processes=num_processes) as pool:
    #        samples = pool.map(self.draw_task, [(topic_ms[i], topic_ws[i]) for i in range(bs)])
    #    return samples
    
    def draw_demon_sequences(self, bs, k):
        topic_ms, topic_ws = self.draw_topics(bs)
        bs_xs = []
        bs_ys = []
        bs_retrieval = []
        bs_learning = []
        for i in range(bs):
            xs = np.random.multivariate_normal(self.new_task_m, self.x_covariance, size=k)
            bs_xs.append(xs)
            retrieval = xs @ self.topic_ws[0]
            learning = xs @ self.new_task_w
            bs_retrieval.append(retrieval)
            bs_learning.append(learning)
            ys = learning[:,np.newaxis] #+ np.random.normal(0, self.y_variance, size=k)
            bs_ys.append(ys)
        bs_xs = np.stack(bs_xs, axis=0)
        bs_ys = np.stack(bs_ys, axis=0)
        bs_retrieval = np.stack(bs_retrieval, axis=0)
        bs_learning = np.stack(bs_learning, axis=0)
        
        return bs_xs, bs_ys, bs_retrieval, bs_learning
    
    def predict(self, xs, ys, target_w=None):
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
        
        adv_m = np.zeros([self.M])
        adv_w = np.zeros([self.M])
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
        
        adv = adv_m+adv_w
        tpis = tpis*scipy.special.softmax(-adv)
        
        tpis = tpis/np.sum(tpis)
        prediction = tpis@(ttopic_ws@xs[-1])
        tw = tpis@ttopic_ws
        
        l_loss = (prediction-ys[-1])**2

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
            'adv_m': adv_m,
            'adv_w': adv_w,
            }
        return return_dict
    
    def visualize(self, fig=None, gs_small=None):
        if self.d not in [2,3]:
            raise Exception("dimension d should be 2 or 3")
        if self.d == 3:
            if gs_small is not None:
                ax1 = fig.add_subplot(gs_small[0, 0], projection='3d')
                ax2 = fig.add_subplot(gs_small[1, 0], projection='3d')
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})
            r = 1.0
            phi = np.linspace(0, np.pi, 100)
            theta = np.linspace(0, 2*np.pi, 100)
            phi, theta = np.meshgrid(phi, theta)
            # Parametric equations for the sphere
            x = r*np.sin(phi)*np.cos(theta)
            y = r*np.sin(phi)*np.sin(theta)
            z = r*np.cos(phi)
            # color for points
            colors = ['#ff8100', '#ffa700']  # Dark Blue to Blue
            n_bins = 100  # Number of bins
            cmap_name = 'custom1'
            custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
            for topic_axs, topic_var, ax, name in [(self.topic_ms, self.sm, ax1, r'$\boldsymbol{\mu}$'), (self.topic_ws, self.sw, ax2, r'$\boldsymbol{w}$')]:
                xs, ys, zs = topic_axs[:,0], topic_axs[:,1], topic_axs[:,2]
                norm_zs = (zs - min(zs)) / (max(zs) - min(zs))
                
                # Plot the wireframe of the sphere for the first subplot
                ax.plot_wireframe(x, y, z, rstride=25, cstride=25, color='b', alpha=0.6, linewidth=0.5)
                # Plot random points on the surface for the first subplot
                scatter = ax.scatter(xs, ys, zs, c=norm_zs, cmap=custom_cmap, s=topic_var*100., label='Prior Center', zorder=5)
                ax.scatter(xs[0], ys[0], zs[0], facecolors='none', edgecolors='black', s=25, label='Target Task', zorder=-5)
                for i, topic_ax in enumerate(topic_axs): # adjust for your actual dimensions
                    ax.text(topic_ax[0], topic_ax[1], topic_ax[2], str(i+1), size=20, zorder=1, color='k')
                # Set properties for the first subplot
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(name)
            
            ax1.scatter(self.new_task_m[0], self.new_task_m[1], self.new_task_m[2], marker='x', c='red', s=25, label='Demo Task', zorder=5)
            ax1.text(self.new_task_m[0], self.new_task_m[1], self.new_task_m[2], str(0), size=20, zorder=1, color='k')
            ax2.scatter(self.new_task_w[0], self.new_task_w[1], self.new_task_w[2], marker='x', c='red', s=25, label='Demo Task', zorder=5)
            ax2.text(self.new_task_w[0], self.new_task_w[1], self.new_task_w[2], str(0), size=20, zorder=1, color='k')
            ax2.legend(loc='upper left', bbox_to_anchor=(-0.35, 1.2))
            #cbar = fig.colorbar(scatter, ax=[ax1, ax2], orientation='vertical', shrink=0.6)
            #cbar.set_label('Depth (Normalized Z-value)')
            
        if self.d == 2:
            if gs_small is not None:
                ax1 = fig.add_subplot(gs_small[0, 0], projection='3d')
                ax2 = fig.add_subplot(gs_small[1, 0], projection='3d')
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})
            r = 1.0
            theta = np.linspace(0, 2*np.pi, 100)
            # Parametric equations for the sphere
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            # color for points
            colors = [(0, 0, 0.2), (0, 0, 1)]  # Dark Blue to Blue
            n_bins = 100  # Number of bins
            
            for topic_axs, topic_var, ax, name in [(self.topic_ms, self.sm, ax1, r'$\boldsymbol{\mu}$'), (self.topic_ws, self.sw, ax2, r'$\boldsymbol{w}$')]:
                xs, ys = topic_axs[:,0], topic_axs[:,1]
                circle = patches.Circle((0, 0), radius=1, fill=False)
                ax.add_patch(circle)
                # Plot random points on the surface for the first subplot
                scatter = ax.scatter(xs, ys, s=topic_var*100, label='Prior Center', zorder=5)
                ax.scatter(xs[0], ys[0], c='lime', marker='x', s=topic_var*100, label='Target Center', zorder=5)
                for i, topic_ax in enumerate(topic_axs):  # adjust for your actual dimensions
                    ax.text(topic_ax[0], topic_ax[1], str(i+1), size=20, zorder=1, color='k')
                # Set properties for the first subplot
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(name)
            
            ax1.scatter(self.new_task_m[0], self.new_task_m[1], c='red', s=25, label='Demo Task', zorder=5)
            ax1.text(self.new_task_m[0], self.new_task_m[1], str(0), size=20, zorder=1, color='k')
            ax2.scatter(self.new_task_w[0], self.new_task_w[1], c='red', s=25, label='Demo Task', zorder=5)
            ax2.text(self.new_task_w[0], self.new_task_w[1], str(0), size=20, zorder=1, color='k')
            ax1.legend()
        if ax is None:
            plt.show()
            
if __name__ == "__main__":
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 40})
    prior = Regular4(sm=0.25, sw=0.25)
    #priorProcesser = PriorProcesser(prior)
    
    prior.visualize()
    plt.tight_layout()
    plt.savefig('prior/3d/regular4.pdf')
    '''
    sampleGenerator = SampleGenerator(3, priorProcesser.new_task_m, priorProcesser.new_task_w, 1)
    
    #sampleGenerator = SampleGenerator(d, new_m_center, new_w_center, 1)
    k_list = []
    retrieval_loss = []
    learning_loss = []
    print('***************')
    K=1000
    max_L = 48
    bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(K, max_L)
    for i in tqdm(range(1,max_L,1)):
        #print()
        r_sum = 0
        l_sum = 0
        k_list.append(i)
        
        #for j in range(K):
        #    xs = bs_xs[j][:i]
        #    ys = bs_xs[j][:i]
        #    retrieval = bs_retrieval[j,i]
        #    learning  = bs_learning [j,i]
        #    #xs,ys = np.ones([1,3]),np.zeros([1,1])
        #    prediction = priorProcesser.predict(xs, ys)['prediction']
        #    r_sum += (prediction-retrieval)**2
        #    l_sum += (prediction-learning)**2
        
        for j in range(K):
            #xs, ys = sampleGenerator.generate_xy_samples(i)
            #xs, ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(1, i)
            xs, ys = bs_xs[j][:i], bs_ys[j][:i]
            retrieval, learning = bs_retrieval[j][i-1], bs_learning[j][i-1]
            
            return_dict = priorProcesser.predict(xs, ys, priorProcesser.topic_ws[0])
            #r_sum += return_dict['r_loss']
            #l_sum += return_dict['l_loss']
            r_sum += (return_dict['prediction'] - retrieval)**2 #return_dict['r_loss']
            l_sum += (return_dict['prediction'] - learning)**2 #return_dict['l_loss']
        retrieval_loss.append(r_sum/K)
        learning_loss.append(l_sum/K)
        #print(return_dict['tw_centers'])
    
    plt.figure(figsize=(16, 4))
    plt.plot(k_list, retrieval_loss, linewidth=4, label='Loss of Task Retrieval')
    plt.plot(k_list, learning_loss,  linewidth=4, label='Loss of Task Learning')
    plt.xlabel('\# of demonstrations', fontsize = 25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize = 25)
    plt.savefig('loss comparison.pdf',bbox_inches='tight')
    '''
