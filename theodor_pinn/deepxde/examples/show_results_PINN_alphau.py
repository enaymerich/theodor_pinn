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
Nwidth = 200 # 100 cells
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
Nheight = 500
dt = simulation_time/Nt

netname = "best"


def animate_net(t):
    time_start = time.perf_counter()
    plot_array = net_output_T(xy_data,t)
    T_net.append(time.perf_counter() - time_start)
    im1_text.set_text(str(round(t, 4)))
    im1.set_array(plot_array.detach().cpu().numpy())
    #im.set_clim(np.min(plot_array.detach().numpy()), np.max(plot_array.detach().numpy()))
    return im1, im1_text


def u_to_t(u, ac0, bc0, tc0):
    b = (ac0 + bc0) * tc0 - u
    c = u*tc0
    return (torch.sqrt(b**2+4*ac0*c) - b) / (2*ac0)

def u_to_t_np(u, ac0, bc0, tc0):
    b = (ac0 + bc0) * tc0 - u
    c = u*tc0
    return (np.sqrt(b**2+4*ac0*c) - b) / (2*ac0)


def t_to_u(t, ac0, bc0, tc0):
    return t*(ac0 +bc0/(1+t/tc0))

def Sample_theo(t):
    tile_meas.u[:,:] = Umin
    for i in range(t):
        for step in range(convergence_steps):
            step_dt = dt / convergence_steps
            tile_meas.solve_backward_surface(step_dt, Tvec)
    Umeas = tile_meas.get_temperature(tile_meas.u)
    HFmeas = tile_meas.get_surface_heatflux()
    return Umeas, HFmeas



def create_pot(x, A=25, sigma=1e-2, mu=0.05, T0=0):
    temp = np.array([A*np.exp(-((x - mu)**2 / (2*sigma**2)))])+T0
    return temp


def net_output_T(xy_data,t):
    time = np.ones(xy_data.shape[0]) * (t/T)
    test_input = np.concatenate((xy_data, time.reshape(-1, 1)), 1)
    # print(test_input[0,:])
    test_input = torch.from_numpy(test_input).float().to('cuda')
    test_predict_HP = my_nn(test_input)*(Umax-Umin)+Umin
    return u_to_t(test_predict_HP.reshape(x.shape[0], y.shape[0]),ac0,bc0,tc0)


def net_output_HF(xy_data,t):
    time = np.ones(xy_data.shape[0]) * (t/T)
    test_input = np.concatenate((xy_data, time.reshape(-1, 1)), 1)
    # print(test_input[0,:])
    test_input = torch.from_numpy(test_input).float().to('cuda')
    test_input.requires_grad = True
    test_predict_HP = my_nn(test_input) * (Umax-Umin)+Umin
    test_predict_HF = -1/Z*dde.grad.jacobian(test_predict_HP, test_input, i=0, j=0)
    return test_predict_HF.reshape(y.shape[0],)

def show_image(fig,ax,data,text=0.0):
    im = ax.imshow(data, vmin=Tmin, vmax=Tmax, extent=[0, 1, 0, 1])
    ax.set_xlabel('poloidal length [m]', fontsize=20)
    ax.set_ylabel('depth [m]', fontsize=20)
    xp = np.linspace(0,1,5)
    ax.set_xticks(xp)
    ax.set_yticks([0.])
    ax.set_xticklabels(np.round(xp*width, 4))
    ax.set_yticklabels(np.round([0.028], 4))
    im_text = plt.text(0.5, 1.01, text, horizontalalignment='center', verticalalignment='bottom',
                       transform=ax.transAxes)
    cbar = fig.colorbar(im)
    return im, im_text, cbar

def show_1D(fig,ax,data, x,text=0.0):
    line = ax.plot(x.reshape(-1,), data.reshape(-1,))
    ax.set_xlabel('profile length [m]', fontsize=16)
    ax.set_ylabel('heat flux [W/m^2]', fontsize=16)
    ticks = ax.get_xticks().tolist()
    ticks[-1] = width
    ax.set_xticks(ticks)
    im_text = plt.text(0.5, 1.01, text, horizontalalignment='center', verticalalignment='bottom',
                       transform=ax.transAxes,fontsize=16)
    return line[0], im_text

layer_size = [3] + [97] * 10 + [1]
activation = "tanh"
initializer = "Glorot uniform"
my_nn = dde.maps.FNN(layer_size, activation, initializer)
#checkpoint = torch.load('../../normtile_fullopt/models/SKOPT_normtile_BC_weak_97_neurons_10_layers_9.4e-05_lr-49000.pt')
#checkpoint = torch.load('../../alphau_fullopt/SKOPT_alphau_BC_weak_26_neurons_8_layers_5.0e-03_lr.pt')

checkpoint = torch.load('./alphau_net_7.pt')
my_nn.load_state_dict(checkpoint['model_state_dict'])
T_net = []
T_theo = []
##Test
my_nn.to('cuda')
x = np.linspace(0, xlim, Nheight)
y = np.linspace(0, width_NN, Nwidth)
xy_data = np.array(np.meshgrid(x, y)).T.reshape(-1,2)
t_data = np.array(np.linspace(0, simulation_time, Nt))
top = np.zeros(y.shape[0])
xy_top = np.concatenate((top.reshape(-1, 1), y.reshape(-1, 1)), 1)
## TILE PROPERTIES
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

qmeas = np.zeros([Nt, Nx])
# qvec = np.zeros([Nt, Nx])

Tmeas = np.zeros([Nt + 1, Nx])  # +1, because initial element is stored as well (t=0)
Tmeas[0] = tile_meas.get_surface_temperature()
# time vector
tvec = np.arange(0, Nt) * dt

fig1, ax1 = plt.subplots()
Unet = net_output_T(xy_data,0.1).detach().cpu().numpy()
HFnet = net_output_HF(xy_top,0.1).detach().cpu().numpy()
im1, im1_text, cbar1 = show_image(fig1, ax1, Unet, '0.1 s')
cbar1.ax.set_ylabel('Tile Temperature [°C]',fontsize=16)
im1.set_clim(Tmin,Tmax)

fig2, ax2 = plt.subplots()
Umeas, HFmeas = Sample_theo(100)
im2, im2_text, cbar2 = show_image(fig2, ax2, Umeas, '0.1 s')
cbar2.ax.set_ylabel('Tile Temperature [°C]',fontsize=16)
im2.set_clim(Tmin,Tmax)

fig3, ax3 = plt.subplots()
im3, im3_text, cbar3 = show_image(fig3, ax3, np.abs(Umeas-Unet), '0.1 s')
cbar3.ax.set_ylabel('Error Temperature [°C]',fontsize=16)
im3.set_clim(0,40)

fig1hf, ax1hf = plt.subplots()
line1, line1_text = show_1D(fig1hf, ax1hf, HFmeas, y*width,'')
line1, line1_text = show_1D(fig1hf, ax1hf, HFnet, y*width,'Heat flux at 0.1 s')
line1, line1_text = show_1D(fig1hf, ax1hf, HFmeas-HFnet, y*width,'')
plt.legend(['THEODOR','PINN','Error'],fontsize=14)
axrel = ax1hf.twinx()
axrel.set_ylim(ax1hf.get_ylim()/np.max(HFmeas)*100)
axrel.set_yticks([0,8,50,100])
axrel.plot(y*width,np.ones_like(y)*8,'r--')
axrel.plot(y*width,np.ones_like(y)*-3,'r--')
axrel.set_ylabel('Heat flux [% of the maximum value]', fontsize=12)

fig4, ax4 = plt.subplots()
Umeas, HFmeas = Sample_theo(1)
im4, im4_text = show_1D(fig4, ax4, Umeas[0,:], y*width, 'Temperature at the surface')
ax4.set_xlim([0,width])
ax4.set_ylabel('Temperature [°C]', fontsize=16)
plt.show()

