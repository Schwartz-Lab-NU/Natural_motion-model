from neo.core import SpikeTrain
from quantities import ms
import numpy as np
import pickle
import quantities as pq
from elephant.spike_train_dissimilarity import victor_purpura_distance
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.special import expit
from scipy.stats import wasserstein_distance
import matplotlib.patches as patches
import h5py
from scipy.signal import convolve
from scipy.interpolate import griddata
import pickle
import pandas as pd
from scipy.stats import ttest_ind

def compute_emd_matrix(spike_trains):
    """
    Compute a pairwise EMD distance matrix for a list of spike trains.
    
    Parameters:
        spike_trains: list of lists, where each sublist is a spike train
    
    Returns:
        emd_matrix: 2D numpy array of shape (N, N) with pairwise EMD values
    """
    num_trials = len(spike_trains)
    emd_matrix = np.zeros((num_trials, num_trials))

    for i in range(num_trials):
        for j in range(i + 1, num_trials):  # Only compute upper triangle
            emd = wasserstein_distance(spike_trains[i], spike_trains[j])
            emd_matrix[i, j] = emd
            emd_matrix[j, i] = emd  # Symmetric matrix

    return emd_matrix

def block_means(matrix,n_groups, group_size):
    # Initialize an array to store the averaged values.
    block_means = np.zeros((n_groups, n_groups))

    # Loop through each block, compute its mean, and store it.
    for i in range(n_groups):
        for j in range(n_groups):
            
            block = matrix[i*group_size:(i+1)*group_size, j*group_size:(j+1)*group_size]
            # Compute the average of the block.
            block_means[i, j] = block.mean()
    return block_means

def spatial_filterSF(rf,A=0.2):
    return A * rf

def gaussian_2d(x, y, sigma_x, sigma_y, A=1, x0=0, y0=0):
    X, Y = np.meshgrid(x, y, indexing='ij')
    return A * np.exp(-((X - x0)**2 / (2 * sigma_x**2) + (Y - y0)**2 / (2 * sigma_y**2)))

def gaussian_2dflatten(x, y, sigma_x, sigma_y,mix, widenratio, A=1, x0=0, y0=0):
    A = gaussian_2d(x=x, y=y, sigma_x = sigma_x, sigma_y = sigma_y, A=A, x0=x0, y0=y0) + gaussian_2d(x=x, y=y, sigma_x = widenratio* sigma_x, sigma_y = widenratio*sigma_y, A=mix*A, x0=x0, y0=y0)
    A = A / np.max(A)
    return A 

def difference_of_gaussians(x, y, sigma_c, sigma_s, A_c=0.4, A_s=0.2):
    X, Y = np.meshgrid(x, y, indexing='ij')
    center = A_c * np.exp(-(X**2 + Y**2) / (2 * sigma_c**2))
    surround = A_s * np.exp(-(X**2 + Y**2) / (2 * sigma_s**2))
    return center - surround

def generalized_gaussian_2d(x, y, sigma_x, sigma_y, alpha, A=1.0, x0=0, y0=0):
    """
    2D generalized Gaussian with exponent alpha:
    - alpha = 1: standard Gaussian
    - 0 < alpha < 1: heavier tails (higher kurtosis)
    - alpha > 1: lighter tails
    """
    X, Y = np.meshgrid(x, y, indexing='ij')
    Q = ((X - x0)**2 / (2 * sigma_x**2) +
         (Y - y0)**2 / (2 * sigma_y**2))
    d = A * np.exp(- Q**alpha)
    d = d / d[int((len(x)-1)/2),int((len(x)-1)/2+1)]
    return d   # Normalize to max value of 1

def student_t_2d(x, y, sigma_x, sigma_y, nu, A=1.0, x0=0, y0=0):
    """
    2D Student's-t distribution (heavy-tailed):
    - nu: degrees of freedom. Lower nu -> heavier tails
    - For nu=1: Cauchy (Lorentzian) form
    """
    X, Y = np.meshgrid(x, y, indexing='ij')
    r2 = ((X - x0)**2 / sigma_x**2 + (Y - y0)**2 / sigma_y**2)
    return A * (1 + r2 / nu) ** (-(nu + 1) / 2)

def biphasic_temporal_filter(t, p1, p2, tau1, tau2, n):
    t = np.asarray(-t, dtype=float)
    term1 = p1 * (t / tau1)**n * np.exp(-n * (t / tau1 - 1))
    term2 = p2 * (t / tau2)**n * np.exp(-n * (t / tau2 - 1))
    return term1 - term2

def apply_temporal_filter_to_movie(movie, temporal_filter):
    F, = movie.shape  
    T = len(temporal_filter) 
    filtered_movie = np.zeros_like(movie)
    filtered_signal = convolve(movie, temporal_filter, mode='full')[:F]
    filtered_movie[:] = filtered_signal
    return filtered_movie

def linear_filter(spatial_rf, noise_input, temporal_filter):   
    T, Y, X = noise_input.shape 
    noise_input = noise_input.astype(np.float32)
    spatial_rf = spatial_rf.astype(np.float32)   
    spatial_filtered_movie = noise_input.reshape((T,Y*X)) @ spatial_rf.reshape((Y*X))  # Shape: (T,)
    filtered_movie = apply_temporal_filter_to_movie(spatial_filtered_movie, temporal_filter)
    return filtered_movie

def apply_nonlinearity(linear_output, method='static', **kwargs):
    if method == 'static':
        return np.maximum(0, linear_output)
    elif method == 'sigmoid':
        gain = kwargs.get('gain', 0.02)
        max_rate = kwargs.get('max_rate', 250.0)
        C = kwargs.get('y', -30)
        return max_rate * expit(gain * (linear_output + C))
    elif method == 'sigmoidgc':
        gain = kwargs.get('gain', 0.05)
        max_rate = kwargs.get('max_rate', 1500.0)
        C = kwargs.get('y', -37256.659)
        return max_rate / (0.04 + np.exp(-gain * (linear_output+165)))  + C
    elif method == 'threshold_linear':
        theta = kwargs.get('theta', 10)
        return np.maximum(0, linear_output - theta)
    elif method == 'threshold_lineargc':
        theta = kwargs.get('theta', 0)
        return np.maximum(0, linear_output - theta)
    elif method == 'powerlaw':
        gamma = kwargs.get('gamma', 0.5)
        return np.maximum(0, np.abs(linear_output)**gamma)
    elif method == 'exponential_saturation':
        alpha = kwargs.get('alpha', 0.1)
        beta = kwargs.get('beta', 1.0)
        return beta * (1 - np.exp(-alpha * linear_output))
    elif method == 'adaptive_gain':
        gain = kwargs.get('gain', 1.0)
        return gain * np.maximum(0, linear_output)
    else:
        raise ValueError("Unknown nonlinearity method")

def resample_and_generate_spikes(firing_rates, target_time_bin=1/1200):
    original_time_bin = 1 / 60
    upsampling_factor = int(original_time_bin / target_time_bin)
    if upsampling_factor < 1:
        raise ValueError("Target time bin must be smaller")
    expanded_firing_rates = np.repeat(firing_rates, upsampling_factor, axis=0)
    scaled_rates = expanded_firing_rates * target_time_bin
    spikes = np.random.poisson(scaled_rates)
    return spikes

def gain_control(Lout, B=0.005, tau=11.0):
    T = len(Lout)  
    v_t = np.zeros(T)  
    g_v = np.zeros(T)
    gain_controlled_Lout = np.zeros(T) 
    gain_controlled_Lout[0] = 1 * Lout[0]
    decay_kernel = B * np.exp(-np.arange(T) / tau)

    for t in range(T):
        v_t[t] = np.sum(gain_controlled_Lout[:t] * decay_kernel[:t][::-1]) 
        g_v[t] = 1 if v_t[t] < 0 else 1 / (1 + v_t[t]**4)
        gain_controlled_Lout[t] = g_v[t] * Lout[t]

    return gain_controlled_Lout, g_v

# Load data
rf_estimated = np.load('/projects/p32750/repository/Natural_motion-model/rf_estimated.npy')

with open('/projects/p32750/repository/Natural_motion-model/spike_indices_n.pkl', 'rb') as f:
    spike_indices_n = pickle.load(f)

with open('/projects/p32750/repository/Natural_motion-model/spike_indices_c.pkl', 'rb') as f:
    spike_indices_c = pickle.load(f)

spike_indicesModified_n = spike_indices_n.copy()
spike_indicesModified_c = spike_indices_c.copy()
spike_indicesModified_n[14] = spike_indices_n[14]-800
spike_indicesModified_n[55] = spike_indices_n[55]-800
spike_indicesModified_c[55] = spike_indices_c[55]-800

# model PARAM
delay = 0.083 # s
# Parameter settting for model
stimuli_n = np.load('/projects/p32750/repository/Natural_motion-model/stimuli_n.npy')
stimuli_c = np.load('/projects/p32750/repository/Natural_motion-model/stimuli_c.npy')
temporal_filter_noise = np.load('/projects/p32750/repository/Natural_motion-model/temporal_filter5.npy')
reversed_temp_rfTemp = temporal_filter_noise[::-1]

# parameters for the LN-GC 
time_para = 5400 # time bin to generate spike
time_bin = 1/time_para
num_modeltrails = 60 #

stimttype = 'control_w/o_gaincontrol'

if stimttype == 'control_w/o_gaincontrol':
    stimuli = stimuli_c
    spike_indices = spike_indicesModified_c
    t_sampling = np.linspace(0,1860,1860)
    theta = 20
    T, X_SF, Y_SF = stimuli.shape 
    x_SF = np.linspace(-(X_SF-1)/2, (X_SF-1)/2, X_SF)
    y_SF = np.linspace(-(Y_SF-1)/2, (Y_SF-1)/2, Y_SF)

spike_trains_n = []
spike_trains_nemd = []

for trial_idx, spikes in enumerate(spike_indices):
    st_n = SpikeTrain(spikes.flatten() / 10 * ms, t_stop=31000)
    st_narray = spikes.flatten() / 10
    spike_trains_n.append(st_n)
    spike_trains_nemd.append(st_narray)

q = 1.0 / (100.0 * pq.ms)

def objectiveVP(params, train=True):
    try:
        gain, max_rate, y, theta = params
        # For control_w/o_gaincontrol stimulus type, we use spatial_filterSF and B=0 (no gain control)
        B = 0  # No gain control
        A = 1  # Fixed A value
        tau = 1  # Fixed tau value for no gain control
        spike_trainGCvpfit = []
        spike_trains_modelvp = []
        T, X_SF, Y_SF = stimuli.shape 
        x_SF = np.linspace(-(X_SF-1)/2, (X_SF-1)/2, X_SF)
        y_SF = np.linspace(-(Y_SF-1)/2, (Y_SF-1)/2, Y_SF)

        # Use spatial_filterSF instead of generalized_gaussian_2d
        spatial_rf = spatial_filterSF(rf=rf_estimated, A=A)
        LoutGCvpfit = linear_filter(spatial_rf, stimuli, reversed_temp_rfTemp)
        # No gain control (B=0)
        LoutGCvpfit, g = gain_control(LoutGCvpfit, B, tau)
        LNoutGCvpfit = apply_nonlinearity(
            apply_nonlinearity(LoutGCvpfit, 'sigmoid', gain=gain, max_rate=max_rate, y=y),
            method='threshold_linear', theta=theta)
        time_bin = 1 / time_para
        if LNoutGCvpfit[0] > 0:
            return 1e10
        # Generate 30 model spike trains.
        for trial in range(30):
            spikingindx = resample_and_generate_spikes(LNoutGCvpfit, time_bin)
            spike_trainGCvpfit.append(np.where(spikingindx > 0)[0])
            st_modelvp = SpikeTrain(np.array(spike_trainGCvpfit[trial]).flatten() / 54 * 10 * ms,t_stop=31000)
            spike_trains_modelvp.append(st_modelvp)
 
        # Select the appropriate real data trials.
        if train:
            real_spike_trains = [spike_trains_n[i] for i in range(len(spike_trains_n)) if i % 2 == 0]
        else:
            real_spike_trains = [spike_trains_n[i] for i in range(len(spike_trains_n)) if i % 2 == 1]
        
        # Combine the real and model-generated spike trains.
        spike_trainsvpfit = real_spike_trains + spike_trains_modelvp
        
        # Check for empty spike trains.
        for st in spike_trains_modelvp:
            if len(st) == 0:
                return 1e10
            if len(st) > 500:
                return 1e10
            if len(st) < 120:    
                return 1e10
        Normalizedrateidx = np.full((len(spike_trainsvpfit),len(spike_trainsvpfit)), np.nan)
        spikecounts = np.array([len(spikes) for spikes in spike_trainsvpfit])
        for i in range(len(spike_trainsvpfit)):
            for j in range(len(spike_trainsvpfit)):
                Normalizedrateidx[i, j] = spikecounts[i] + spikecounts[j]

        # Compute the Victor-Purpura distance and block means.
        vp_matrix = victor_purpura_distance(spike_trainsvpfit, q)
        vp_f_norm = vp_matrix / Normalizedrateidx
        scoresvp = block_means(vp_f_norm, 2, 30)
        scorevp = scoresvp[0, 1]
        return scorevp

    except Exception as e:
        print("Error in objectiveVP with params", params, ":", e)
        return 1e10

# Define bounds for A, tau, gain, max_rate, y (B=0 for no gain control)
bounds = [
          (0.001, 0.01),   # gain
          (150, 300),      # max_rate
          (-600, 400),
          (10, 100)]      # theta

# Prepare lists to store iteration data.
iterations = []
errors_train = []
errors_eval = []

def callbackF(xk, convergence):
    # Record current iteration number.
    iteration = len(iterations) + 1
    # Compute errors on training and evaluation sets.
    train_error = objectiveVP(xk, train=True)
    eval_error = objectiveVP(xk, train=False)
    iterations.append(iteration)
    errors_train.append(train_error)
    errors_eval.append(eval_error)
    print(f"Iteration {iteration}: Training Error = {train_error}, Evaluation Error = {eval_error}")

def objectiveVP_wrapper(params):
    return objectiveVP(params, train=True)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # Only needed if you're freezing the code into an executable; safe to include otherwise.

    result_de = differential_evolution(
        objectiveVP_wrapper,
        bounds,
        strategy="rand2bin",
        maxiter=400,
        mutation=(0.5, 1),
        recombination=0.95,
        disp=True,
        callback=callbackF,
        workers=-1
    )

    optimal_gain, optimal_max, optimal_y, optimal_theta = result_de.x

    print("Optimal parameters found:")

    print("gain    =", optimal_gain)
    print("max_rate =", optimal_max)
    print("y =", optimal_y)
    print("theta =", optimal_theta)

    optimal_params = result_de.x

    # Save the outputs to a file for later inspection
    results = {
        "optimal_params": optimal_params,
        "iterationsC": iterations,
        "errors_trainC": errors_train,
        "errors_evalC": errors_eval
    }

    with open("/projects/p32750/repository/Natural_motion-model/V1_control_w_o_gaincontrol_VPN.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Optimization results saved to V1_control_w_o_gaincontrol_VPN.pkl") 