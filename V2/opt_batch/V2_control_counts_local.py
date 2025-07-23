import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import HTMLWriter
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import convolve
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
import pickle
import stimgenerate as sg
import matplotlib.patches as mpatches
import pandas as pd
import scipy.stats as stats
import matplotlib.patches as patches
from neo.core import SpikeTrain
from quantities import ms
import quantities as pq
from elephant.spike_train_dissimilarity import victor_purpura_distance
from scipy.optimize import differential_evolution
from scipy.special import expit
from scipy.stats import wasserstein_distance
from scipy.integrate import simpson
import random
import time
import cProfile
import pstats
from functools import wraps
from multiprocessing import shared_memory
from functools import partial
import os

def timing_decorator(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        return result
    return wrapper


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

@timing_decorator
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

@timing_decorator
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
# with open('spike_indices_nV2.pkl', 'rb') as f:
#     spike_indices_n = pickle.load(f)
with open('spike_indices_cV2.pkl', 'rb') as f:
    spike_indices_c = pickle.load(f)
rfC = np.load('rfC_V2.npy')
rfC[rfC < 0.1] = 0
spatial_filterV2 = rfC

# model PARAM
delay = 0.083 # s
t_sampling = np.linspace(0,151,151)

time_para = 5400 # time bin to generate spike
time_bin = 1/time_para
num_modeltrails = 60 #
t_temporal = np.linspace(-2000/3,0, 40) 
q = 1.0 / (100.0 * pq.ms)

random.seed(42)
all_indices = list(range(700))
random.shuffle(all_indices)
train_indices = all_indices[:350]
n =3
eval_indices = all_indices[350:]

# spike_trains_n = []
# spike_trains_nemd = []
# for trial_idx, spikes in enumerate(spike_indices_n):
#     st_n = SpikeTrain(spikes.flatten() / 10 * ms, t_stop=15200/6)
#     st_narray = spikes.flatten() / 10
#     spike_trains_n.append(st_n)
#     spike_trains_nemd.append(st_narray)

spike_trains_c = []
spike_trains_cemd = []
for trial_idx, spikes in enumerate(spike_indices_c):
    st_c = SpikeTrain(spikes.flatten() / 10 * ms, t_stop=15200/6)
    st_carray = spikes.flatten() / 10
    spike_trains_c.append(st_c)
    spike_trains_cemd.append(st_carray)



class Profiler:
    """Simple profiler to track timing of different steps"""
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def start(self, step_name):
        """Start timing a step"""
        self.start_times[step_name] = time.time()
    
    def end(self, step_name):
        """End timing a step and record the duration"""
        if step_name in self.start_times:
            duration = time.time() - self.start_times[step_name]
            if step_name in self.timings:
                self.timings[step_name].append(duration)
            else:
                self.timings[step_name] = [duration]

    
    def get_average_timings(self):
        """Get average timing for each step"""
        return {step: np.mean(times) for step, times in self.timings.items()}
    
    def print_summary(self):
        """Print timing summary"""
        print("\n=== TIMING SUMMARY ===")
        for step, times in self.timings.items():
            avg_time = np.mean(times)
            total_time = np.sum(times)
            count = len(times)
            print(f"{step}: avg={avg_time:.4f}s, total={total_time:.4f}s, count={count}")


# Global profiler instance
profiler = Profiler()


def objectiveCounts_random(params, train=True, npy_path=None, shape=None, dtype=None):
    # Use np.memmap to access the stimuli array from file
    profiler.start("stimuli_loading")
    try:
        movies_c = np.load(npy_path, mmap_mode='r')
    except Exception as e:
        print("Error in objectiveCounts_random with params", params, ":", e)
        return 1e10
    profiler.end("stimuli_loading")

    try:
        profiler.start("total_objective")
        
        profiler.start("parameter_unpacking")
        B, tau, p1, p2, tau1, tau2,gain,max_rate,y,theta  = params
        profiler.end("parameter_unpacking")

        profiler.start("spatial_filter_setup")
        spatial_rf = spatial_filterV2
        profiler.end("spatial_filter_setup")
        
        profiler.start("temporal_filter_computation")
        reversed_temp_rf = biphasic_temporal_filter(t_temporal, p1, p2, tau1, tau2, n)[::-1]
        profiler.end("temporal_filter_computation")
        
        # Generate model spike trains.
        spike_trainGCvpfit = []
        spike_trains_modelvp = []

        profiler.start("index_selection")
        if train:
            selected_indices = train_indices

        else:
            selected_indices = eval_indices

        profiler.end("index_selection")

        profiler.start("reference_spike_trains")
        reference_spike_trains = [spike_trains_c[i] for i in selected_indices]
        profiler.end("reference_spike_trains")


        
        profiler.start("model_computation_loop")
        for i in selected_indices:
            profiler.start("single_trial_computation")
            
            profiler.start("stimuli_extraction")
            stimuli = movies_c[i]
            profiler.end("stimuli_extraction")
            
            profiler.start("linear_filter")
            LoutGCvpfit = linear_filter(spatial_rf, stimuli, reversed_temp_rf)
            profiler.end("linear_filter")
            
            profiler.start("gain_control")
            LoutGCvpfit, g = gain_control(LoutGCvpfit, B, tau)
            profiler.end("gain_control")
            
            profiler.start("nonlinearity")
            LNoutGCvpfit = apply_nonlinearity(
                apply_nonlinearity(LoutGCvpfit, 'sigmoid', gain=gain, max_rate=max_rate, y=y),
                method='threshold_linear', theta=theta)
            profiler.end("nonlinearity")
            
            if LNoutGCvpfit[0] > 0:
                return 1e10
                
            profiler.start("spike_count_calculation")
            spike_trains_modelvp.append(np.mean(LNoutGCvpfit) * 151 / 60)
            profiler.end("spike_count_calculation")
            
            profiler.end("single_trial_computation")
        profiler.end("model_computation_loop")

        profiler.start("distance_calculation")
        spike_distances_vp = np.zeros(len(spike_trains_modelvp))
        for i in range(len(spike_trains_modelvp)):
            if reference_spike_trains[i].size == 0:
                Counts = 0
            else:
                Counts = reference_spike_trains[i].shape[0]

            spike_distances_vp[i] = abs(Counts - spike_trains_modelvp[i])

        score = np.nanmean(spike_distances_vp)
        profiler.end("distance_calculation")
        print(score)
        profiler.end("total_objective")

        return score

    except Exception as e:
        print("Error in objectiveCounts_random with params", params, ":", e)
        return 1e10
             
bounds = [(0, 0.05), (0, 25), (1, 2), (0.1,1.5), (20,80),(80,200),(0.0001, 0.01), (50,400), (-900,4000),(5,140)] # Nonlinearity(0.001, 0.008), (220,380), (-900,-200),(5,140)

# Prepare lists to store iteration data.
iterations = []
errors_train = []
errors_eval = []

def callbackF(xk, convergence, npy_path, shape, dtype):
    # Record current iteration number.
    iteration = len(iterations) + 1
    # Compute errors on training and evaluation sets.
    profiler.start("callback_evaluation")
    train_error = objectiveCounts_random(xk, train=True, npy_path=npy_path, shape=shape, dtype=dtype)
    eval_error = objectiveCounts_random(xk, train=False, npy_path=npy_path, shape=shape, dtype=dtype)
    profiler.end("callback_evaluation")
    iterations.append(iteration)
    errors_train.append(train_error)
    errors_eval.append(eval_error)
    print(f"Iteration {iteration}: Training Error = {train_error}, Evaluation Error = {eval_error}")

def objective_wrapper(params):
    return objectiveCounts_random(params, train=True)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    profiler.start("stimuli_loading")
    # Use np.load with mmap_mode to avoid loading all data into RAM
    npy_path = 'V2_stimuli_c.npy'
    movies_c_np = np.load(npy_path, mmap_mode='r')
    shape = movies_c_np.shape
    dtype = movies_c_np.dtype
    profiler.end("stimuli_loading")
    # Prepare partial function for multiprocessing
    worker_func = partial(objectiveCounts_random, npy_path=npy_path, shape=shape, dtype=dtype)
    callback_func = partial(callbackF, npy_path=npy_path, shape=shape, dtype=dtype)
    profiler.start("total_optimization")
    result_de = differential_evolution(
        worker_func,
        bounds,
        strategy="rand2bin",
        popsize=18,
        maxiter=1,
        mutation=(0.5, 1),
        recombination=0.95,
        disp=True,
        callback=callback_func,
        workers=-1,
        polish=False
    )
    profiler.end("total_optimization")

    optimal_B, optimal_tau, optimal_1, optimal_2,optimal_3,optimal_4,optimal_5,optimal_6,optimal_7,optimal_8 = result_de.x
    optimal_params = result_de.x
    print("Optimal parameters found:")
    print("B       =", optimal_B)
    print("tau     =", optimal_tau)
    print("p1      =", optimal_1)
    print("p2      =", optimal_2)
    print("tau1    =", optimal_3)
    print("tau2    =", optimal_4)
    print("gain    =", optimal_5)
    print("max_rate=", optimal_6)
    print("y       =", optimal_7)
    print("theta   =", optimal_8)

    # Print timing summary
    profiler.print_summary()

    # Save the outputs to a file for later inspection
    results = {
        "optimal_params": optimal_params,
        "iterationsC": iterations,
        "errors_trainC": errors_train,
        "errors_evalC": errors_eval,
        "timing_summary": profiler.get_average_timings(),
        "detailed_timings": profiler.timings
    }

    with open("V2_control_counts_test.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Optimization results saved to V2_control_counts_test.pkl") 