import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import deepxde as dde
import time


# Backend pytorch
import torch

# Bring your packages onto the path
import sys, os
path = '../../../../Dottorato/NN PDE/theo-pytorch/theo-pytorch/theodor'
sys.path.append(os.path.abspath(path))
# Backend theodor
# for 1D Surface information use 2D solver
import THEODOR_2D
import materials
import matplotlib 
font = {'family' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font) 
# Example tile parameters:
width = 560e-3# 30e-3 30 mm wide
height = 28e-3 # 5 mm height
## params for Theodor
convergence_steps = 1

# positive integer for Nheight defined the number of layers.
# negative values define the CFL number used, for which the
# layer count depends on the sample rate (recommended value ~1-3)

Z = 560e-3
T = 0.1
height_NN = 1#5e-3
width_NN = 1#28e-3
tile_thick = 28e-3
tile_maxwidth = Z
tile_minwidth = 100e-3
mu = width/2
sigma = 1e-1
Time_size = 1
Z = tile_maxwidth
simulation_time = 0.1
Net_time = simulation_time/T

ac0 = 19.086175786524592
bc0 = 146.4990242134754
tc0 = 1088.804568376853
Tmax = 2000#K
Tmin = 25#°C
Umax = (Tmax)*(ac0 +bc0/(1+ (Tmax)/tc0))
Umin = (Tmin)*(ac0 +bc0/(1+ (Tmin)/tc0))
# size of the tile
xlim = tile_thick / Z

## Computation grid
Nt= 100
Nwidth = 200
Nheights = np.round(np.linspace(10,1000,15),0)
dt = simulation_time/Nt

netname = "_"

def u_to_t_np(u, ac0, bc0, tc0):
    b = (ac0 + bc0) * tc0 - u
    c = u*tc0
    return (np.sqrt(b**2+4*ac0*c) - b) / (2*ac0)


def create_pot(x, A=25, sigma=1e-2, mu=0.05, T0=0):
    temp = np.array([A*np.exp(-((x - mu)**2 / (2*sigma**2)))])+T0
    return temp

layer_size = [3] + [97] * 10 + [1]
activation = "tanh"
initializer = "Glorot uniform"
my_nn = dde.maps.FNN(layer_size, activation, initializer)
#checkpoint = torch.load('../../normtile_fullopt/models/SKOPT_normtile_BC_weak_97_neurons_10_layers_9.4e-05_lr-49000.pt')
#checkpoint = torch.load('../../alphau_fullopt/SKOPT_alphau_BC_weak_26_neurons_8_layers_5.0e-03_lr.pt')

checkpoint = torch.load('./alphau_net_7.pt')
my_nn.load_state_dict(checkpoint['model_state_dict'])
##Test
my_nn.to('cuda')
my_nn.eval()
T_net_avg = []
T_net_min = []
T_net_max = []
T_theo_avg = []
T_theo_max = []
T_theo_min = []
for Nheight in Nheights:
    T_net = []
    T_theo = []
    Nheight = int(Nheight)
    print('Number of points', Nheight)
    x = np.linspace(0, xlim, Nheight)
    y = np.linspace(0, width_NN, Nwidth)
    xy_data = np.array(np.meshgrid(x, y)).T.reshape(-1,2)
    t_data = np.array(np.linspace(0, simulation_time, Nt))
    top = np.zeros(y.shape[0])
    xy_top = np.concatenate((top.reshape(-1, 1), y.reshape(-1, 1)), 1)
    ims = []
    for t in t_data:
        time_array = np.ones(xy_data.shape[0]) * (t/T)
        test_input = np.concatenate((xy_data, time_array.reshape(-1, 1)), 1)
        # print(test_input[0,:])
        test_input = torch.from_numpy(test_input).float().to('cuda')
        time_start = time.perf_counter()
        test_predict_HP = my_nn(test_input)*(Umax-Umin)+Umin
        T_net.append(time.perf_counter()-time_start)
    print('elapsed time NET:', np.mean(T_net),'min',np.min(T_net),'max',np.max(T_net))
    
    
    mat = materials.material(materialname='divertor_OP12')
    ### THEODOR
    tile_meas = THEODOR_2D.diffusion_2D(
        width = width,
        Nwidth = Nwidth,
        height = height,
        Nheight = Nheight,
        ref_dt = dt,
        T0 = Tmin,
        material = mat)
    # the heat flux data are 2D - 1D in space and 1D in time.
    Nx = Nwidth
    Umeas = np.zeros([Nheight, Nx, Nt])
    
    xx, yy = np.meshgrid(x, y)
    sigma = 1e-1
    y_tile = np.linspace(0, width, Nwidth)
    Uvec = create_pot(y_tile, A=(Umax-Umin), sigma=sigma*width, mu=width/2, T0=Umin)
    Tvec = u_to_t_np(Uvec, ac0, bc0, tc0)
    tile_meas.sample_surface_temperature(Tvec)
    Umeas = tile_meas.get_temperature(tile_meas.u)
    for t in t_data:
      Umeas = tile_meas.get_temperature(tile_meas.u)
      time_start = time.perf_counter()    
      for i in range(convergence_steps):
          step_dt = dt / convergence_steps
          tile_meas.solve_backward_surface(step_dt, Tvec) #se varia Tvec in superficie va cambiato codice
      T_theo.append(time.perf_counter() - time_start)
    print('elapsed time THEO:', np.mean(T_theo),'min',np.min(T_theo),'max',np.max(T_theo))
    T_net_avg.append(np.mean(T_net))
    T_net_min.append(np.min(T_net))
    T_net_max.append(np.max(T_net))    
    T_theo_avg.append(np.mean(T_theo))
    T_theo_min.append(np.min(T_theo))
    T_theo_max.append(np.max(T_theo))


plt.figure()
plt.semilogy(Nheights*Nwidth, T_net_avg,'b-')
plt.semilogy(Nheights*Nwidth, T_theo_avg,'r-')
#plt.semilogy(Nheights*Nwidth, T_net_min,'b--')
#plt.semilogy(Nheights*Nwidth, T_net_max,'b--')
#plt.semilogy(Nheights*Nwidth, T_theo_min,'r--')
#plt.semilogy(Nheights*Nwidth, T_theo_max,'r--')
plt.legend(['PINN','THEO'],fontsize=16)
plt.xlabel('number of points', fontsize=20)
plt.ylabel('elapsed time [s]', fontsize=20)