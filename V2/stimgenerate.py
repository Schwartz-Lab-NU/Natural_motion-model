
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import HTMLWriter


def generate_stimuli(trajx, trajy,board_size=400, diameter=30,fps = 60,pretime = 50,tailtime =50,type='natural',preframes=15,tailframes=45,stimframes=15):
    stimuli = []
    dt = 1/fps

    board_range = [-(board_size/2), (board_size/2)]  
    radius = diameter / 2  

    x_vals = np.linspace(board_range[0], board_range[1], board_size+1)
    y_vals = np.linspace(board_range[0], board_range[1], board_size+1)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    if type == 'natural':
        if len(trajx) != len(trajy):
            raise ValueError("trajx and trajy must have the same length.")
        if len(trajx) == 0:
            raise ValueError("Trajectory arrays must not be empty.")

        preframenum = int(pretime / 1000 /dt)
        tailframenum = int(tailtime / 1000 /dt)
        for _ in range(preframenum):
            frame = np.zeros((board_size+1, board_size+1), dtype=np.uint8)
            stimuli.append(frame)   
        
        for frame_idx, (x, y) in enumerate(zip(trajx, trajy)):
            # if frame_idx < preframenum or frame_idx >= (len(trajx) - tailframenum):
            #     # Draw blank
            #     frame = np.zeros((board_size, board_size), dtype=np.uint8)
            # else:
                # Draw the circle
            frame = np.zeros((board_size+1, board_size+1), dtype=np.uint8)
            mask = (X - x) ** 2 + (Y - y) ** 2 <= radius ** 2
            frame[mask] = 1

            stimuli.append(frame)   
            
        for _ in range(tailframenum):
            frame = np.zeros((board_size+1, board_size+1), dtype=np.uint8)
            stimuli.append(frame)   
    if type == 'spotfield':
        if len(trajx) != len(trajy):
            raise ValueError("trajx and trajy must have the same length.")
        if len(trajx) == 0:
            raise ValueError("Trajectory arrays must not be empty.")

    
        for section_idx, (x, y) in enumerate(zip(trajx, trajy)):
            frame = np.zeros((board_size+1, board_size+1), dtype=np.uint8)
            for _ in range(preframes):
                stimuli.append(frame)   
                
            frame = np.zeros((board_size+1, board_size+1), dtype=np.uint8)
            mask = (X - x) ** 2 + (Y - y) ** 2 <= radius ** 2
            frame[mask] = 1
            for _ in range(stimframes):
                stimuli.append(frame)   
            frame = np.zeros((board_size+1, board_size+1), dtype=np.uint8)
            for _ in range(tailframes):
                stimuli.append(frame)   
        
    return np.array(stimuli)

def STA(T, spike_trainLN, traj_feature):
    all_vn, all_an, all_r_n, trajx_n, trajy_n, dt, time_para = traj_feature
    d_n_model_LN = np.zeros((T, len(spike_trainLN), max([len(spikes[0]) for spikes in spike_trainLN])))
    sta_v_n_model_LN = np.full((T, len(spike_trainLN), max([len(spikes[0]) for spikes in spike_trainLN])), np.nan)
    sta_a_n_model_LN = np.full((T, len(spike_trainLN), max([len(spikes[0]) for spikes in spike_trainLN])), np.nan)
    sta_x_n_model_LN = np.full((T, len(spike_trainLN), max([len(spikes[0]) for spikes in spike_trainLN])), np.nan)
    sta_y_n_model_LN = np.full((T, len(spike_trainLN), max([len(spikes[0]) for spikes in spike_trainLN])), np.nan)
    sta_r_n_model_LN = np.full((T, len(spike_trainLN), max([len(spikes[0]) for spikes in spike_trainLN])), np.nan)
    for trial_idx, spikes in enumerate(spike_trainLN):
        spike_indices = (spikes[0].flatten() / time_para / dt ).astype(int)
        for spike_idx, si in enumerate(spike_indices):
            i0 = si - T
            sta_v_n_model_LN[max(-i0 + 1, 0):, trial_idx, spike_idx] = all_vn[trial_idx][max(i0 - 1, 0):max(si - 1, 0)]
            sta_a_n_model_LN[max(-i0 + 2, 0):, trial_idx, spike_idx] = all_an[trial_idx][max(i0 - 2, 0):max(si - 2, 0)]
            sta_x_n_model_LN[max(-i0, 0):, trial_idx, spike_idx] = trajx_n[trial_idx][max(i0, 0):si]
            sta_y_n_model_LN[max(-i0, 0):, trial_idx, spike_idx] = trajy_n[trial_idx][max(i0, 0):si]
            sta_r_n_model_LN[max(-i0, 0):, trial_idx, spike_idx] = all_r_n[trial_idx][max(i0, 0):si]
            d_n_model_LN[max(-i0, 0):, trial_idx, spike_idx] += 1
    sta_v_n_model_LN /= d_n_model_LN
    sta_a_n_model_LN /= d_n_model_LN
    sta_x_n_model_LN /= d_n_model_LN
    sta_y_n_model_LN /= d_n_model_LN
    sta_r_n_model_LN /= d_n_model_LN
    avg_sta_v_n_model_LN = np.nanmean(sta_v_n_model_LN, axis=(1, 2))
    avg_sta_a_n_model_LN = np.nanmean(sta_a_n_model_LN, axis=(1, 2))
    avg_sta_x_n_model_LN = np.nanmean(sta_x_n_model_LN, axis=(1, 2))
    avg_sta_y_n_model_LN = np.nanmean(sta_y_n_model_LN, axis=(1, 2))
    avg_sta_r_n_model_LN = np.nanmean(sta_r_n_model_LN, axis=(1, 2))
    time_window = np.linspace(-T * dt, -dt, T)
    sem_x_n_model_LN = np.nanstd(sta_x_n_model_LN.reshape(T, -1), axis=1) / np.sqrt(np.nansum(d_n_model_LN, axis=(1,2)))
    sem_y_n_model_LN = np.nanstd(sta_y_n_model_LN.reshape(T, -1), axis=1) / np.sqrt(np.nansum(d_n_model_LN, axis=(1,2)))
    sem_r_n_model_LN = np.nanstd(sta_r_n_model_LN.reshape(T, -1), axis=1) / np.sqrt(np.nansum(d_n_model_LN, axis=(1,2)))
    sem_v_n_model_LN = np.nanstd(sta_v_n_model_LN.reshape(T, -1), axis=1) / np.sqrt(np.nansum(d_n_model_LN, axis=(1,2)))
    sem_a_n_model_LN = np.nanstd(sta_a_n_model_LN.reshape(T, -1), axis=1) / np.sqrt(np.nansum(d_n_model_LN, axis=(1,2)))
    return time_window, avg_sta_v_n_model_LN, avg_sta_a_n_model_LN, avg_sta_x_n_model_LN, avg_sta_y_n_model_LN, avg_sta_r_n_model_LN, sem_x_n_model_LN, sem_y_n_model_LN, sem_r_n_model_LN, sem_v_n_model_LN, sem_a_n_model_LN

def animate_stimuli(stimuli, x_range, y_range, filename="stimuli_animation.html"):

    fig, ax = plt.subplots(figsize=(6, 6))
    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    im = ax.imshow(stimuli[0], extent=extent, origin='lower')
    ax.set_title("Stimuli Animation")
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    cbar = plt.colorbar(im, ax=ax, label="Stimulus Intensity")
    
    def update(frame):
        im.set_data(stimuli[frame])
        ax.set_title(f"Stimulus Frame {frame + 1}")
        return im,

    ani = FuncAnimation(fig, update, frames=stimuli.shape[0], interval=16.67, blit=True)
    ani.save(filename, writer=HTMLWriter())
    print(f"Animation saved as {filename}")
    plt.show()

    return ani

def animate_LNout(LNout, x_range, y_range, filename="stimuli_animation.html"):

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(x_range[0], x_range[1], LNout.shape[1]+1)
    y = np.linspace(y_range[0], y_range[1], LNout.shape[2]+1)
    X, Y = np.meshgrid(x, y)

    vmin, vmax = LNout.min(), LNout.max()

    surf = [ax.plot_surface(X, Y, LNout[0,:, :], cmap='plasma', vmin=vmin, vmax=vmax)]
    ax.set_zlim(vmin, vmax)
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title("Stimulus Frame 1")

    def update(frame):

        surf[0].remove()
        surf[0] = ax.plot_surface(X, Y, LNout[frame,:, :], cmap='plasma', vmin=vmin, vmax=vmax)
        ax.set_title(f"Stimulus Frame {frame + 1}")
        return surf[0],

    ani = FuncAnimation(fig, update, frames=LNout.shape[0], interval=16.67, blit=True)
    ani.save(filename, writer=HTMLWriter())
    print(f"Animation saved as {filename}")
    plt.show()

    return ani

