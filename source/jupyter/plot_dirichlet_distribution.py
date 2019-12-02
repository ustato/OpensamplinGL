# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # OpenGLを用いたMCMCのリアルタイム可視化

# OpenGLとGLFWをインポートします
from OpenGL.GL import *
from OpenGL.GLUT import *
import glfw
import numpy as np
from math import gamma


class gibbs_random_chain():
    def __init__(self, a, *, rho=0.5, diagonal_components=1):
        self.a = np.array(a)
        self.K = self.a.size
        self.rho = rho
        self.SIGMA = (np.ones((self.K,self.K))-np.eye(self.K))*rho + np.eye(self.K)*diagonal_components
        self.L = np.linalg.cholesky(self.SIGMA)
        self.z_t =  self.L.dot(self.produce_vec_y())
        self.z_t = np.ones(self.K)/self.K
        
    # 区間(-1,1)の一様乱数ベクトルzを利用して独立なD個のサンプルベクトルyを得る関数
    def produce_vec_y(self):
        z_array = 2*np.random.rand(self.K) - 1
        while (z_array**2).sum() > 1:
            z_array = 2*np.random.rand(self.K) - 1
        squared_r = (z_array**2).sum()
        y = z_array * np.sqrt((-2*np.log(squared_r)/squared_r))
        self.z_array = z_array
        return y
    
    def produce_z_k(self, z_not_k):
        z_k_limit = 1 - (z_not_k**2).sum()
        return np.random.rand()*np.sqrt(z_k_limit)*(2*np.random.randint(0,2)-1)
    
    def generate_random(self):
        z_array = np.array(self.z_array)
        ind = np.ones(self.K, dtype=bool)
        for k in range(self.K):
            ind[k] = False
            z_array[k] = self.produce_z_k(z_array[ind])
            ind[k] = True
        return self.z_t + self.L.dot(z_array), z_array

    def set_z_t(self, z_t):
        self.z_t = z_t
        
    def set_z_array(self, z_array):
        self.z_array = z_array

    def alpha(self, z):
        x = np.abs(np.array(z))
        x /= x.sum()
        x_t = np.abs(np.array(self.z_t))
        x_t /= x_t.sum()
        return min(1, p_tilde(x,self.a)/p_tilde(x_t,self.a))


class gibbs_over_relaxation():
    def __init__(self, a, *, rho=0, diagonal_components=1, param_mu=0, param_sigma=1, param_alpha=0):
        self.a = np.array(a)
        self.K = self.a.size
        self.rho = rho
        self.SIGMA = (np.ones((self.K,self.K))-np.eye(self.K))*rho + np.eye(self.K)*diagonal_components
        self.inv_SIGMA = np.linalg.inv(self.SIGMA)
        self.L = np.linalg.cholesky(self.SIGMA)
        self.z_t =  self.L.dot(self.produce_vec_y())
        self.z_t = np.ones(self.K)/self.K
        if (type(param_mu) is int) or (type(param_mu) is float): 
            self.param_mu = np.ones(self.K)*param_mu
        else:
            self.param_mu = np.array(param_mu)
        if (type(param_sigma) is int) or (type(param_sigma) is float): 
            self.param_sigma = np.ones(self.K)*param_sigma
        else:
            self.param_sigma = np.array(param_sigma)
        if (type(param_alpha) is int) or (type(param_alpha) is float): 
            self.param_alpha = param_alpha
        else:
            raise ValueError("param_alpha is needed scalar")
        
    # 区間(-1,1)の一様乱数ベクトルzを利用して独立なD個のサンプルベクトルyを得る関数
    def produce_vec_y(self):
        z_array = 2*np.random.rand(self.K) - 1
        while (z_array**2).sum() > 1:
            z_array = 2*np.random.rand(self.K) - 1
        squared_r = (z_array**2).sum()
        y = z_array * np.sqrt((-2*np.log(squared_r)/squared_r))
        self.z_array = z_array
        return y
    
    def produce_z_k(self, z_not_k):
        z_k_limit = 1 - (z_not_k**2).sum()
        return np.random.rand()*np.sqrt(z_k_limit)*(2*np.random.randint(0,2)-1)
    
    def generate_random(self):
        z_array = np.array(self.z_array)
        ind = np.ones(self.K, dtype=bool)
        for k in range(self.K):
            ind[k] = False
            z_array[k] = self.produce_z_k(z_array[ind])
            ind[k] = True
        z_dash = (self.param_mu + self.param_alpha*(self.z_t-self.param_mu) + (self.param_sigma**2)*((1-self.param_alpha**2)**(1/2))*z_array)
        return z_dash, z_array

    def set_z_t(self, z_t):
        self.z_t = z_t
        
    def set_z_array(self, z_array):
        self.z_array = z_array

    def alpha(self, z):
        x = np.abs(np.array(z))
        x_t = np.abs(np.array(self.z_t))
        x /= x.sum()
        x_t /= x_t.sum()
        return min(1, p_tilde(x,self.a)/p_tilde(x_t,self.a)*np.exp(((x-self.param_mu).dot(self.inv_SIGMA).dot(x-self.param_mu)-(x_t-self.param_mu).dot(self.inv_SIGMA).dot(x_t-self.param_mu))/2))


def draw():
    global mean_point
    global sigma_points
    
    glClearColor(0.0, 0.5, 0.5, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 軸をプロット
    glColor(1.0, 1.0, 1.0, 1.0)
    glBegin(GL_TRIANGLES)
    glVertex(p1)
    glVertex(p2)
    glVertex(p3)
    glEnd()
    
    # データ点をプロット
    glPointSize(5)
    glColor(0.7, 0.7, 0.2, 0.0)
    glBegin(GL_POINTS)
    for x in x_accepted:
        glVertex(x)
    glEnd()
    
    # 分散をプロット
    n_sigma = len(sigma_points)
    colormap = np.linspace(0,1,n_sigma)
    for n in range(n_sigma)[::-1]:
        glColor(colormap[n_sigma-1-n], 0.0, colormap[n], 0.0)
        glBegin(GL_TRIANGLES)
        for x in sigma_points[n]:
            glVertex(x)
        glEnd()
        
    # 重心をプロット
    glPointSize(10)
    glColor(0.0, 0.0, 0.0, 0.0)
    glBegin(GL_POINTS)
    glVertex(mean_point)
    glEnd()
    
    # 軸の説明文
    glRasterPos(p1[0]-0.1, p1[1])
    [glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(s)) for s in "theta_1"]
    glRasterPos(p2)
    [glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(s)) for s in "theta_2"]
    glRasterPos(p3[0]-0.2, p3[1])
    [glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(s)) for s in "theta_3"]
    
    # 操作説明
    glRasterPos(p1[0]-1.0, p1[1]+0.2)
    [glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(s)) for s in "                 q:quit"]
    glRasterPos(p1[0]-1.0, p1[1]+0.1)
    [glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(s)) for s in "number key:run n times"]
    
    glRasterPos(p1[0]+0.3, p1[1]+0.2)
    [glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(s)) for s in "black dot:means"]
    glRasterPos(p1[0]+0.3, p1[1]+0.1)
    [glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(s)) for s in "green dot:samples"]
    glRasterPos(p1[0]+0.3, p1[1])
    [glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(s)) for s in "red      triangle:1 sigma area"]
    glRasterPos(p1[0]+0.3, p1[1]-0.1)
    [glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(s)) for s in "purple triangle:2 sigma area"]
    glRasterPos(p1[0]+0.3, p1[1]-0.2)
    [glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(s)) for s in "blue    triangle:3 sigma area"]
    
    glFlush()
    glutSwapBuffers()


def resize(w, h):
    glViewport(0, 0, w, h)


# +
# def mouse(button, state, x, y):    
#     print(x, y)

# +
# def motion(x, y):
#     print("drag:", x, y)
# -

def keyboard(key, x, y, *, n_samples=10):
    global mean_point
    global sigma_points
    
    key = key.decode('utf-8')
    if key == 'q':
        sys.exit()
    elif key.isdigit():
        n_samples = int(key)
        for i in range(n_samples):
            z_dash,z_array = mcmc.generate_random()
            mcmc.set_z_t(z_dash)
            mcmc.set_z_array(z_array)
            x_dash = np.array(z_dash)
            x_dash += np.abs(x_dash.min())*2
            x_dash /= x_dash.sum()
            z_accepted.append(z_dash)
            x_accepted.append(p1 * x_dash[0] + p2 * x_dash[1] + p3 * x_dash[2])
            
        mean_z_dash = np.mean(z_accepted, axis=0)
        mean_point = mean_z_dash + np.abs(mean_z_dash.min())*2
        mean_point /= mean_point.sum()
        mean_point = p1 * mean_point[0] + p2 * mean_point[1] + p3 * mean_point[2]
        
        var_z_dash = np.var(z_accepted, axis=0)
        sigma_points = [np.array([mean_z_dash for i in mean_z_dash]) for n in range(3)]
        for n in range(3):
            for i in range(mean_z_dash.size):
                sigma_points[n][i][i] += (n+1) * var_z_dash[i]
                sigma_points[n][i] += np.abs(sigma_points[n][i].min())*2
                sigma_points[n][i] /= sigma_points[n][i].sum()
        sigma_points = [[p1 * s[0] + p2 * s[1] + p3 * s[2] for s in sigma] for sigma in sigma_points]


        # 描画を更新
        glutPostRedisplay()
#         print(np.cov(np.array(z_accepted).T))

if __name__ == '__main__':
    # ディリクレ分布の各次元とする頂点を指定
    p1 = np.array([0.0, (3.0**0.5) - 1.0])
    p2 = np.array([-1.0, -1.0])
    p3 = np.array([1.0, -1.0])

    # ディリクレ分布
    p = lambda x_vec,a: (gamma(np.sum(a))/np.prod([gamma(a_k) for a_k in a])) * np.prod(np.power(x_vec, np.array(a)-1))
    
    # サンプリングが容易な目標分布
    p_tilde = lambda x_vec,a: np.prod(np.power(x_vec, np.array(a)-1))

    # ディリクレ分布のパラメータ
    dir_a = np.array([10, 10, 0.5])

    # サンプル数
    n_burn_in = 5000

#     mcmc = gibbs_random_chain(dir_a, rho=0.9)
    mcmc = gibbs_over_relaxation(dir_a, param_mu=dir_a, param_sigma=2, param_alpha=0.9)

    # 初期化処理のバーンイン期間を実行
    for n in range(n_burn_in):
        z_dash,z_array = mcmc.generate_random()
        u = np.random.rand()
        if mcmc.alpha(z_dash) >= u:
            mcmc.set_z_t(z_dash)
            mcmc.set_z_array(z_array)

    # サンプリング結果を格納するリスト
    z_accepted = []
    x_accepted = []
    mean_z_dash = np.ones(dir_a.size)
    mean_z_dash /= mean_z_dash.sum()
    mean_point = p1 * mean_z_dash[0] + p2 * mean_z_dash[1] + p3 * mean_z_dash[2]
    var_z_dash = np.ones(dir_a.size)
    sigma_points = [np.array([mean_z_dash for i in mean_z_dash]) for n in range(3)]
    for n in range(3):
        for i in range(mean_z_dash.size):
            sigma_points[n][i][i] += (n+1) * var_z_dash[i]
            sigma_points[n][i] += np.abs(sigma_points[n][i].min())*2
            sigma_points[n][i] /= sigma_points[n][i].sum()
    sigma_points = [[p1 * s[0] + p2 * s[1] + p3 * s[2] for s in sigma] for sigma in sigma_points]
    
    # OpenGLの処理内容をセット  
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutCreateWindow("Dirichlet distribution 3 dims to 2D")
    glutReshapeFunc(resize)
    glutDisplayFunc(draw)
    # glutMouseFunc(mouse)
    # glutMotionFunc(motion)
    glutKeyboardFunc(keyboard)

    # 描画開始
    glutMainLoop()

