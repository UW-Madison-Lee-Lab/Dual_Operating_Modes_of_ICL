import numpy as np
from multiprocessing import Pool
import time
class Prior:
    def __init__(self, d, M, topic_ms, topic_ws, pis, s, t, sm, sw):
        self.d = d
        self.M = M
        self.topic_ms = topic_ms.copy()
        self.topic_ws = topic_ws.copy()
        self.pis = pis
        self.s = s
        self.x_covariance = self.s**2 * np.eye(self.d)
        self.t = t
        self.y_variance = self.t**2
        self.sm = sm
        self.m_covariance = self.sm**2 * np.eye(self.d)
        self.sw = sw
        self.w_covariance = self.sw**2 * np.eye(self.d)
        self.delta_m = (self.sm/self.s)**2
        self.delta_w = (self.sw/self.t)**2
    
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
    def arxiv_draw_sequences(self, bs, k, num_processes=12):
        topic_ms, topic_ws = self.draw_topics(bs)
        samples = []
        with Pool(processes=num_processes) as pool:
            samples = pool.map(self.draw_task, [(topic_ms[i], topic_ws[i]) for i in range(bs)])
        return samples
    '''
        # Choose a component based on weights
        index = np.random.choice(self.M, p=self.pis)
        
        # Draw a sample from the chosen component
        xs = np.random.multivariate_normal(self.m_centers[index], self.m_covariance, size=k+1)
        ys = xs@self.w_centers[index]
        
        return xs, ys
    '''
    
if __name__ == "__main__":
    # set up prior
    d = 3 # dimension
    M = 6 # number of topic
    q = 1 # different ratio between different pi
    min_distance = 0.05
    # generate centers
    print('generate m_centers')
    m_centers = np.array([[+1,0,0],[0,+1,0],[0,0,+1],
                          [-1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
    print('generate w_centers')
    w_centers = np.array([[+1,0,0],[0,+1,0],[0,0,+1],
                          [-1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
    print('generate pis')
    pis = np.ones([6])/6
    
    s = 1 # sigma
    t = 1 # tau
    sm = 0.1 # sigma_mu
    sw = 0.1 # sigma_w
    
    prior = Prior(d, M, m_centers, w_centers, pis, s, t, sm, sw)
    bs_xs, bs_ys, bs_zs = prior.draw_sequences(bs=256, k=32)
    a = time.time()
    bs_xs = np.stack(bs_xs, axis=0)
    bs_ys = np.stack(bs_ys, axis=0)
    e = time.time()
    print(e-a)
    
    
    
    
    