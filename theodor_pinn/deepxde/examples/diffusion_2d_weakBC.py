"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import random
# Backend pytorch
import torch
import matplotlib.pyplot as plt
from torch import nn

##Fix seed
seed = 155
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled= False
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
# size of the tile
height = 1#5e-3
width = 1#28e-3
mu = width/2
sigma = 1e-1
alpha = 1
#decay_steps = 1e3
#decay_rate = 1e-4
Time_size = 0.1#height*width/C

## Domain par
ND = 10000# Nx*Ny*Nt
Nx = int((ND/alpha*height/width)**(1/4))#20#
Ny = int(Nx*width/height)
Nt = int(Nx*Ny*alpha*Time_size/width/height)
Nsx = Ny*Nt#1000
Nsy = Nx*Nt#2000
Nst = Nx*Ny#100

num_test = 5000
## Network parameters
lr = 3e-4
loss = 'MSE'
optim = "adam"
## weights
weights = [0.25,0.75,0.75,0.75]
#weights = np.ndarray.tolist(np.exp(weights)/sum(np.exp(weights)))
#weights = [0, 0, 0, 1]
logname = 'theo_deepxde_BC_weak_alpha075_earlystop'


def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=2)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    # Backend pytorch
    return dy_t - alpha * (dy_xx+dy_yy)#((dy_t.squeeze() - C * (dy_xx+dy_yy).squeeze())* torch.exp(-x[:,2])).unsqueeze(1)


def boundary_x(x, on_boundary):
    return on_boundary & np.isclose(x[0], 0)#qui xlim

def boundary_y(x, on_boundary):
    return (on_boundary & np.isclose(x[1], 0)) or (on_boundary & np.isclose(x[1], width))


def func_x(x):
    return np.exp(-(mu-x[:, 1:2])**2/(2*sigma**2))


def func_y(x):
    return x[:, 1:2]*0


def func_IC(x):
    return np.exp(-(mu-x[:, 1:2])**2/(2*sigma**2))*np.exp(-(x[:, 0:1])**2/(2*sigma**2))


def show_train_points(data):
    train_points = data.train_points()
    bc_points = data.bc_points()
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    SC1 = ax1.scatter(train_points[:, 2], train_points[:, 0], train_points[:, 1])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('t')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    bc_vals = [func_x(bc_points[i:i+1,:]) if bc_points[i,0]==0 else 0 for i in range(len(bc_points))]
    SC2 = ax2.scatter(bc_points[:,2], bc_points[:,0], bc_points[:,1], c=bc_vals)
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('y')
    plt.colorbar(SC2)
    plt.show()
    return

## mesh geometry
geom = dde.geometry.Rectangle([0, 0], [height,width])
## time space
timedomain = dde.geometry.TimeDomain(0, Time_size)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

#bc1 = dde.DirichletBC(geomtime, func, lambda _, on_boundary: )
bc_x = dde.DirichletBC(geomtime, func_x, boundary_x, component=0, num_points=Nsx)#Nsx)
bc_y = dde.NeumannBC(geomtime, func_y, boundary_y, component=0, num_points=Nsy*2)#)

#)
ic = dde.IC(geomtime, func_IC, lambda _, on_initial: on_initial)

#np.array(np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1]],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]))
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_x, bc_y, ic],
    num_domain=ND,  #ND
    num_boundary=(Nsx+Nsy)*2, #Nx+Ny
    num_initial=Nst,  #Nt
    num_test=num_test,
    train_distribution="pseudo",
    seed=seed,
 #   exclusions=
)
#show_train_points(data)

layer_size = [3] + [32] * 2 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
#for i in range(1, len(layer_sizes)):
#    net.linears.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
#    initializer(net.linears[-1].weight)
#    initializer_zero(net.linears[-1].bias)
if net.linears[-1].bias is not None:
     nn.init.constant_(net.linears[-1].bias.data, 0.5)


model = dde.Model(data, net)
loss_names=['PDE_loss', 'BC_x loss','BC_y loss','IC loss']
model.compile(optim, lr=lr, loss=loss, loss_weights=weights)
resampler = dde.callbacks.PDEResidualResampler(period=10000)
early_stop = dde.callbacks.EarlyStopping(min_delta=1e-8, patience=50000)
losshistory, train_state = model.train(epochs=500000, display_every=500, save_every=5000,
                                       model_save_path=logname+'.pt',
                                       #model_restore_path=logname+'.pt',
                                       logname=logname,
                                       callbacks=[resampler, early_stop],
                                       Tensorboard=True, loss_names=loss_names)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
dde.save_loss_history(losshistory,logname+'.txt')