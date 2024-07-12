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

# Example tile parameters:
width = 560e-3# 30e-3 30 mm wide
Nwidth = 200# 100 cells
height = 28e-3 # 5 mm height
## params for Theodor
convergence_steps = 1
width_NN = 1
height_NN = 1
# positive integer for Nheight defined the number of layers.
# negative values define the CFL number used, for which the
# layer count depends on the sample rate (recommended value ~1-3)
K = 100 #W/K
D = 70e-6 #m^2/s
simulation_time = 0.1
Nheight = 100
Nt = 100
Z = width/width_NN
T = 0.1/Z**2
Net_time = simulation_time/T
xlim = (height/width)
dt = simulation_time/Nt
min_temp = 0
max_temp = 1

def animate_net(t):
    time_start = time.perf_counter()
    with torch.no_grad():
        plot_array = net_output_HP(xy_data,t)
    T_net.append(time.perf_counter() - time_start)
    im1_text.set_text(str(round(t, 4)))
    im1.set_array(plot_array.detach().cpu().numpy())
    #im.set_clim(np.min(plot_array.detach().numpy()), np.max(plot_array.detach().numpy()))
    return im1, im1_text

def animate_net_hf(t):
    plot_array = net_output_HF(xy_top, t)
    im_text.set_text(str(round(t, 4)))
    line.set_ydata(plot_array.detach().cpu().numpy())
    # im.set_clim(np.min(plot_array.detach().numpy()), np.max(plot_array.detach().numpy()))
    return line, im_text


def animate_theo(t):

    Umeas = tile_meas.u
    im2.set_data(Umeas)
    time_start = time.perf_counter()    
    for i in range(convergence_steps):
        step_dt = dt / convergence_steps
        tile_meas.solve_backward_surface(step_dt, Tvec) #se varia Tvec in superficie va cambiato codice
    T_theo.append(time.perf_counter() - time_start)
    im2_text.set_text(str(round(t*dt,4)))
    return im2, im2_text

def animate_theo_hf(t):
    HFmeas = tile_meas.get_surface_heatflux()
    line.set_ydata(HFmeas.reshape(-1,))
    for i in range(convergence_steps):
        step_dt = dt / convergence_steps
        tile_meas.solve_backward_surface(step_dt, Tvec) #se varia Tvec in superficie va cambiato codice
    im_text.set_text(str(round(t*dt,4)))
    return line, im_text


def animate_error(t):

    Umeas = tile_meas.u
    with torch.no_grad():
        plot_array = net_output_HP(xy_data, t*dt)
    # print(plot_array.detach().numpy()[:,0])
    error = np.abs(Umeas-plot_array.detach().cpu().numpy())
    im3.set_data(error)
    im_text.set_text(str(round(t*dt,4)))
    for i in range(convergence_steps):
        step_dt = dt / convergence_steps
        tile_meas.solve_backward_surface(step_dt, Tvec) #se varia Tvec in superficie va cambiato codice
    return im3, im_text

def animate_error_hf(t):

    HFmeas = tile_meas.get_surface_heatflux()
    plot_array = net_output_HF(xy_top,t*dt)
    # print(plot_array.detach().numpy()[:,0])
    error = np.abs(HFmeas-plot_array.detach().cpu().numpy())
    print('error %', max(error)/max(HFmeas)*100)
    line.set_ydata(error)
    im_text.set_text(str(round(t*dt,4)))
    for i in range(convergence_steps):
        step_dt = dt / convergence_steps
        tile_meas.solve_backward_surface(step_dt, Tvec) #se varia Tvec in superficie va cambiato codice
    return line, im_text

def create_temp(x, A=25, sigma=1e-2, mu=0.05, T0=0):
    temp = np.array([A*np.exp(-((x - mu)**2 / (2*sigma**2)))])+T0
    return temp


def net_output_HP(xy_data,t):
    time = np.ones(xy_data.shape[0]) * (t/T)
    test_input = np.concatenate((xy_data, time.reshape(-1, 1)), 1)
    # print(test_input[0,:])
    test_input = torch.from_numpy(test_input).float().to('cuda')
    test_predict_HP = my_nn(test_input) * (max_temp - min_temp) + min_temp
    return test_predict_HP.reshape(x.shape[0], y.shape[0])


def net_output_HF(xy_data,t):
    time = np.ones(xy_data.shape[0]) * (t/T)
    test_input = np.concatenate((xy_data, time.reshape(-1, 1)), 1)
    # print(test_input[0,:])
    test_input = torch.from_numpy(test_input).float().to('cuda')
    test_input.requires_grad = True
    test_predict_HP = my_nn(test_input) * (max_temp - min_temp) + min_temp
    test_predict_HF = -1/Z*dde.grad.jacobian(test_predict_HP, test_input, i=0, j=0)
    return test_predict_HF.reshape(y.shape[0],)

def show_image(fig,ax,data):
    im = ax.imshow(data, vmin=min_temp, vmax=max_temp, extent=[0, 1, 0, 1])
    ax.set_xlabel('y', fontsize=20)
    ax.set_ylabel('x', fontsize=20)
    ax.set_xticklabels(np.round(ax1.get_xticks() * width, 4))
    ax.set_yticklabels(np.round(ax1.get_yticks()[-1:] * height, 4))
    im_text = plt.text(0.5, 1.01, 0.0, horizontalalignment='center', verticalalignment='bottom',
                       transform=ax1.transAxes)
    cbar = fig.colorbar(im)
    return im, im_text, cbar

def show_1D(fig,ax,data, x):
    line = ax.plot(x.reshape(-1,), data.reshape(-1,))
    ax.set_xlabel('profile length', fontsize=20)
    ax.set_ylabel('heat flux', fontsize=20)
    ax.set_xticklabels(np.round(ax1.get_xticks() * width, 4))
    im_text = plt.text(0.5, 1.01, 0.0, horizontalalignment='center', verticalalignment='bottom',
                       transform=ax1.transAxes)
    return line[0], im_text

layer_size = [3] + [97] * 10 + [1]
netname = "97_neurons_10_layers"
activation = "tanh"
initializer = "Glorot uniform"
my_nn = dde.maps.FNN(layer_size, activation, initializer)
checkpoint = torch.load('../../normtile_fullopt/models/SKOPT_normtile_BC_weak_97_neurons_10_layers_9.4e-05_lr-49000.pt')
#checkpoint = torch.load('../../normtile_fullopt/models/SKOPT_normtile_BC_weak_32_neurons_2_layers_3.0e-04_lr-83000.pt')
my_nn.load_state_dict(checkpoint['model_state_dict'])
my_nn.eval()
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
ims = []
# First set up the figure, the axis, and the plot element we want to animate
fig1, ax1 = plt.subplots()
iternum = 0
plot_array = net_output_HP(xy_data,t_data[0]).detach().cpu()
im1, im1_text, cbar1 = show_image(fig1,ax1,plot_array)
cbar1.ax.set_ylabel('Tile heat potential [a.u.]')
im1.set_clim(0,max_temp)
# call the animator.  blit=True means only re-draw the parts that have changed.
#


anim = animation.FuncAnimation(fig1, animate_net, t_data, interval=50, blit=False, repeat_delay=1000)
f = r"../../animations/animation_" + netname + "_norm_net.gif"
writergif = animation.PillowWriter(fps=15)
anim.save(f, writer=writergif)
print('elapsed time NET:', np.mean(T_net),'min',np.min(T_net),'max',np.max(T_net))
fig_hf1, ax_hf1 = plt.subplots()
iternum = 0
plot_array = net_output_HF(xy_top,t_data[0]).detach().cpu()
line, im_text = show_1D(fig_hf1,ax_hf1,plot_array, y)
ax_hf1.set_ylim(0,2e5)
# call the animator.  blit=True means only re-draw the parts that have changed.
#
anim = animation.FuncAnimation(fig_hf1, animate_net_hf, t_data, interval=50, blit=False, repeat_delay=1000)
f = r"../../animations/animation_" + netname + "_norm_net_heatflux.gif"
writergif = animation.PillowWriter(fps=15)
anim.save(f, writer=writergif)

### THEODOR
tile_meas = THEODOR_2D.diffusion_2D(
    width = width,
    Nwidth = Nwidth,
    height = height,
    Nheight = Nheight,
    ref_dt = dt,
    T0 = min_temp,
    material = None,
    D=D,
    K=K)
# the heat flux data are 2D - 1D in space and 1D in time.
Nx = Nwidth
Umeas = np.zeros([Nheight, Nx, Nt])

xx, yy = np.meshgrid(x, y)
sigma = 1e-1
y_tile = np.linspace(0, width, Nwidth)
Tvec = create_temp(y_tile, A=max_temp-min_temp, sigma=sigma*width, mu=width/2, T0=min_temp)/K
tile_meas.sample_surface_temperature(Tvec)
Umeas = tile_meas.get_temperature(tile_meas.u)

qmeas = np.zeros([Nt, Nx])
# qvec = np.zeros([Nt, Nx])

# resulting temperature
# Tvec = np.zeros([Nt+1, Nx]) # +1, because initial element is stored as well (t=0)
Tmeas = np.zeros([Nt + 1, Nx])  # +1, because initial element is stored as well (t=0)
# Tvec[0] = tile_ref.get_surface_temperature()
Tmeas[0] = tile_meas.get_surface_temperature()
# time vector
tvec = np.arange(0, Nt) * dt

fig2, ax2 = plt.subplots()
im2, im2_text, cbar2 = show_image(fig2, ax2, Umeas)
cbar2.ax.set_ylabel('Tile heat potential [a.u.]')
im2.set_clim(0,max_temp)
anim = animation.FuncAnimation(fig2, animate_theo, [i for i in range(Nt)], interval=50, blit=False, repeat_delay=1000)
f = r"../../animations/theodor_norm.gif"
writergif = animation.PillowWriter(fps=15)
anim.save(f, writer=writergif)
print('elapsed time THEO:', np.mean(T_theo),'min',np.min(T_theo),'max',np.max(T_theo))


tile_meas.u = tile_meas.u*0
fig_hf2, ax_hf2 = plt.subplots()
iternum = 0
tile_meas.sample_surface_temperature(Tvec)
HFmeas = tile_meas.get_surface_heatflux()
line, im_text = show_1D(fig_hf2,ax_hf2, HFmeas, y)
ax_hf2.set_ylim(0,2e5)

anim = animation.FuncAnimation(fig_hf2, animate_theo_hf, [i for i in range(Nt)], interval=50, blit=False, repeat_delay=1000)
f = r"../../animations/animation_" + netname + "_theo_heatflux.gif"
writergif = animation.PillowWriter(fps=15)
anim.save(f, writer=writergif)

tile_meas.u = tile_meas.u*0
plot_array = net_output_HP(xy_data,t_data[0])[0].detach().cpu().numpy()
fig3, ax3 = plt.subplots()
im3, im_text, cbar3 = show_image(fig3, ax3, abs(Umeas-plot_array))
im3.set_clim(0,0.1)
cbar3.ax.set_ylabel('Error [a.u.]')
anim = animation.FuncAnimation(fig3, animate_error, [i for i in range(Nt)], interval=50, blit=False, repeat_delay=1000)
f = r"../../animations/animation_" + netname + "_norm_error.gif"
writergif = animation.PillowWriter(fps=15)
anim.save(f, writer=writergif)

tile_meas.u = tile_meas.u*0
fig_hf3, ax_hf3 = plt.subplots()
plot_array = net_output_HF(xy_top,t_data[0]).detach().cpu().numpy()
iternum = 0
tile_meas.sample_surface_temperature(Tvec)
HFmeas = tile_meas.get_surface_heatflux()
line, im_text = show_1D(fig_hf3,ax_hf3, HFmeas-plot_array, y)
ax_hf3.set_ylim(0,2e4)
anim = animation.FuncAnimation(fig_hf3, animate_error_hf, [i for i in range(Nt+1)], interval=50, blit=False, repeat_delay=1000)
f = r"../../animations/animation_" + netname + "_error_heatflux.gif"
writergif = animation.PillowWriter(fps=15)
fig_hf3.canvas.draw()
anim.event_source.stop()
anim.save(f, writer=writergif)

