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
# %% functions
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
    profiler.start("linear_filter_setup")
    # spatial_rf = spatial_rf.astype(np.float32)
    # noise_input = noise_input.astype(np.float32)  
    profiler.end("linear_filter_setup")
    profiler.start("spatial_filter_computation")
    spatial_filtered_movie = noise_input.reshape((T,Y*X)) @ spatial_rf.reshape((Y*X))  # Shape: (T,)
    profiler.end("spatial_filter_computation")
    profiler.start("temporal_filter_computation")
    filtered_movie = apply_temporal_filter_to_movie(spatial_filtered_movie, temporal_filter)
    profiler.end("temporal_filter_computation")
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
# Duplicate gaussian_2d definition removed to avoid override.

# %% load data

# with open('spike_indices_nV2.pkl', 'rb') as f:
#     spike_indices_n = pickle.load(f)
with open('/projects/p32750/repository/Natural_motion-model/V3/results/spike_indices_cc2V3.pkl', 'rb') as f:
    spike_indices_c = pickle.load(f)
X_SF = 401
Y_SF = 401
A = 1

x_SF = np.linspace(-(X_SF-1)/2, (X_SF-1)/2, X_SF)
y_SF = np.linspace(-(Y_SF-1)/2, (Y_SF-1)/2, Y_SF)

# model PARAM
delay = 0.05 # s
t_sampling = np.linspace(0,92,93)

time_para = 5400 # time bin to generate spike
time_bin = 1/time_para

t_temporal = np.linspace(-2000/3,0, 40) 
q = 1.0 / (100.0 * pq.ms)

random.seed(42) #answer to life the universe and everything
all_indices = list(range(682))
random.shuffle(all_indices)
train_indices = all_indices[:340]
n =3
eval_indices = all_indices[340:]

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
    st_c = SpikeTrain(spikes.flatten() / 10 * ms, t_stop=1550)
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


def objectiveCounts_random(params, train=True):
    
    # Use np.memmap to access the stimuli array from file
    profiler.start("Linear_loading")
    global GLOBAL_LINEAR 
    Lall = GLOBAL_LINEAR
    profiler.end("Linear_loading")

    try:
        profiler.start("total_objective")
        
        profiler.start("parameter_unpacking")
        B, tau, p1, p2, tau1, tau2,gain,max_rate,y,theta  = params
        profiler.end("parameter_unpacking")

        #profiler.start("spatial_filter_setup")


        
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
            
            #profiler.start("stimuli_extraction")
            #stimuli = 
            #profiler.end("stimuli_extraction")
            
            #profiler.start("linear_filter")
            #LoutGCvpfit = linear_filter(spatial_rf, stimuli, reversed_temp_rf)
            #profiler.end("linear_filter")
            LoutGCvpfit =  apply_temporal_filter_to_movie(Lall[i],reversed_temp_rf)
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
            spike_trains_modelvp.append(np.mean(LNoutGCvpfit) * 92 / 60)
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

        profiler.end("total_objective")

        return score

    except Exception as e:
        print("Error in objectiveCounts_random with params", params, ":", e)
        return 1e10
             
bounds = [(0, 0.05), (0.8, 25), (1, 2), (0.1,1.5), (20,80),(80,200),(0.0001, 0.01), (50,400), (-900,4000),(5,140)] # Nonlinearity(0.001, 0.008), (220,380), (-900,-200),(5,140)

def sigma_objective(sigma_array):
    sigma = sigma_array[0]
    print(f"Evaluating sigma = {sigma:.3f}")

    # Generate RF and GLOBAL_LINEAR
    profiler.start("stimuli_loading")
    # Precompute spatial filtering
    with h5py.File('/projects/p32750/repository/Natural_motion-model/V3/results/stimulic2_cV3.h5', 'r') as f:
        movies_f32 = f['stimuli'][:]
        movies_f32 = movies_f32.astype(np.float32, copy=False)
    n_trials, T, Y, X = movies_f32.shape
    movies_2d = movies_f32.reshape(n_trials * T, Y * X)
    spatial_rf = gaussian_2d(x_SF, y_SF, sigma, sigma, A=1, x0=0, y0=0)
    spatial_rf_flat = spatial_rf.reshape(Y*X).astype(np.float32)
    linear_flat = movies_2d @ spatial_rf_flat
    global GLOBAL_LINEAR
    GLOBAL_LINEAR = linear_flat.reshape(n_trials, T)
    del movies_f32, movies_2d, linear_flat
    profiler.end("stimuli_loading")
        
    # Initialize iteration tracking for this sigma evaluation
    iterations = []
    errors_train = []
    errors_eval = []
    
    def callbackF(xk, convergence):
        # Record current iteration number.
        iteration = len(iterations) + 1
        # Compute errors on training and evaluation sets.
        profiler.start("callback_evaluation")
        train_error = objectiveCounts_random(xk, train=True)
        eval_error = objectiveCounts_random(xk, train=False)
        profiler.end("callback_evaluation")
        iterations.append(iteration)
        errors_train.append(train_error)
        errors_eval.append(eval_error)
        print(f"Iteration {iteration}: Training Error = {train_error}, Evaluation Error = {eval_error}")
    
    profiler.start("DE_optimization")
    # Run DE on other params
    result_de = differential_evolution(
        objectiveCounts_random,
        bounds,
        strategy="rand2bin",
        popsize=16,
        maxiter=400,
        mutation=(0.5, 1),
        recombination=0.95,
        disp=False,
        callback=callbackF,
        workers=-1,
        polish=False
    )
    profiler.end("DE_optimization")
    optimal_B, optimal_tau, optimal_1, optimal_2,optimal_3,optimal_4,optimal_5,optimal_6,optimal_7,optimal_8 = result_de.x
    current_score = result_de.fun
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
    print(f"Score for sigma={sigma:.2f}: {current_score:.4f}")

    # Only update global best if this is better
    global BEST_DE_PARAMS, BEST_DE_SCORE, BEST_SIGMA, BEST_ITERATIONS, BEST_ERRORS_TRAIN, BEST_ERRORS_EVAL
    if BEST_DE_SCORE is None or current_score < BEST_DE_SCORE:
        BEST_DE_PARAMS = result_de.x
        BEST_DE_SCORE = current_score
        BEST_SIGMA = sigma
        BEST_ITERATIONS = iterations.copy()
        BEST_ERRORS_TRAIN = errors_train.copy()
        BEST_ERRORS_EVAL = errors_eval.copy()
        print(f"*** NEW BEST SCORE: {current_score:.4f} with sigma={sigma:.3f} ***")

    return current_score


def objective_wrapper(params):
    return objectiveCounts_random(params, train=True)
import os

def get_unique_rss_mb():
    """
    Returns the unique (private) resident set size, in MB,
    by summing Private_Clean + Private_Dirty from /proc/[pid]/smaps.
    """
    pid = os.getpid()
    total_kb = 0
    try:
        with open(f'/proc/{pid}/smaps', 'r') as f:
            for line in f:
                if line.startswith('Private_Clean:') or line.startswith('Private_Dirty:'):
                    # lines look like 'Private_Clean:     1234 kB'
                    parts = line.split()
                    total_kb += int(parts[1])
        return total_kb / 1024.0
    except FileNotFoundError:
        return None  # not on Linux or /proc inaccessible
# %%
if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('fork', force=True)  
    from multiprocessing import freeze_support
    freeze_support()

    # Initialize global variables
    global GLOBAL_LINEAR
    global BEST_DE_PARAMS
    global BEST_DE_SCORE
    global BEST_SIGMA
    GLOBAL_LINEAR = None
    BEST_DE_PARAMS = None
    BEST_DE_SCORE  = None
    BEST_SIGMA = None
    sigmas = np.linspace(5, 150, 30)   # 1-unit steps from 10 to 100
    scores = np.zeros_like(sigmas)
    # evaluate

    # Prepare partial function for multiprocessing
    profiler.start("total_optimization")
    from scipy.optimize import minimize
    for i, s in enumerate(sigmas):
        scores[i] = sigma_objective([s])

    print(f"\nBest sigma: {sigmas[np.argmin(scores)]:.3f} with score: {np.min(scores):.4f}")
    np.savez("V3_control_counts_sigmas.npz", sigmas=sigmas, scores=scores)
    # res = minimize(
    #     sigma_objective,
    #     x0=[30.0],                     # Initial guess for σ
    #     bounds=[(10.0, 100.0)],        # Adjust bounds as needed
    #     method='L-BFGS-B',
    #     options={'disp': True, 'maxiter': 10}
    # )

    # print(f"\nBest sigma: {res.x[0]:.3f} with score: {res.fun:.4f}")
    profiler.end("total_optimization")
    print("best σ:",        BEST_SIGMA)
    print("best DE score:", BEST_DE_SCORE)
    print("best DE params:", BEST_DE_PARAMS)


    if BEST_DE_PARAMS is not None:
        optimal_B, optimal_tau, optimal_1, optimal_2,optimal_3,optimal_4,optimal_5,optimal_6,optimal_7,optimal_8 = BEST_DE_PARAMS
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
    uniq = get_unique_rss_mb()
    if uniq is not None:
        print(f"Unique (private) RSS: {uniq:.1f} MB")
    # Save the outputs to a file for later inspection
    results = {
        "best_sigma": BEST_SIGMA,
        "optimal_params": BEST_DE_PARAMS,
        "iterationsC": BEST_ITERATIONS,
        "errors_trainC": BEST_ERRORS_TRAIN,
        "errors_evalC": BEST_ERRORS_EVAL,
        "timing_summary": profiler.get_average_timings(),
        "detailed_timings": profiler.timings
    }

    with open("/projects/p32750/repository/Natural_motion-model/V3/results/V3_control_counts_gaussian.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Optimization results saved to V3_control_counts_gaussian.pkl") 




