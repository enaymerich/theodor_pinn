import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate(fig, fun_anim, iterable, f, interval=50, blit=False, rep_del=1000, fps=15):
    anim = animation.FuncAnimation(fig, fun_anim, iterable, interval=interval, blit=blit, repeat_delay=rep_del)
    writergif = animation.PillowWriter(fps=fps)
    anim.save(f, writer=writergif)
    return anim