# Third party imports
from random import sample
import numpy as np
import rpy2.robjects as robjects
from scipy.signal import find_peaks, hilbert

# Local imports
from src.visualization.data_vis import *

def partition_spikes(spike_data):
    
    neuron_labels = spike_data[:, 0] # The first column indicates numeric neuron label
    partition_indices = np.diff(neuron_labels) # Intermediate variable
    
    partition_indices = np.where(partition_indices != 0)[0] + 1 # Incrementing label indicates partition
    
    return partition_indices

def separate_neuron_data(partition_indices, spike_data):

    separated_spike_data = []

    for i, partition in enumerate(partition_indices, start=0):

        if i == 0:
            spike_times = spike_data[:partition, 1]
        else:
            spike_times = spike_data[partition_indices[i - 1]:partition, 1]

        separated_spike_data.append(spike_times)

    spike_times = spike_data[partition:, 1]
    separated_spike_data.append(spike_times)

    return separated_spike_data

def r_spike_kde(aligned_spike_times: list, bandwidth, N: int, N_range: list) -> tuple:
    """Use rpy2 to compute Gaussian KDE on aligned spike times.

    Args:
        aligned_spike_times: List of 1D numpy arrays containing all spike 
            times across trials.
        bandwidth: The bandwidth of the kernel used to compute KDE
        N: The number of linearly-spaced points on the domain of 
            aligned_spike_times at which to compute the probability density
            estimate.
        N_range: List where len(N_range) = 2 specifying left and right 
            endpoints of interval on which to compute KDE. 
    Returns:
        kde_x: A list of 1D numpy arrays containing domain on which KDE was 
            computed for each neuron (index-matched).
        kde_y: A list of 1D numpy arrays containing probability density 
            estimation over d_x for each neuron (index-matched).
    """

    kde_x = []
    kde_y = []
    for spike_times in aligned_spike_times:
        r_density = robjects.r['density']
        kde = r_density(robjects.FloatVector(spike_times), bw=bandwidth, n=N, **{'from':N_range[0], 'to':N_range[1]})
        kde_x.append(np.array(kde[0]))
        kde_y.append(np.array(kde[1]))

    return kde_x, kde_y

def dominant_frequency_exp(kde_x: np.ndarray, kde_y: np.ndarray, plot=False) -> tuple:

    """Compute dominant KDE frequency in time and frequency domains and optionally plot results.

    Args:
        kde_x: The domain over which KDE was computed.
        kde_y: The probability density estimate over kde_x.
    Returns:
        dominant_freq_fft: Global maximum in frequency domain exluding DC offset.
        dominant_freq_time: KDE frequency estimated by dividing number of maxima by signal
            duration.
            
    """
    # Define constants 
    N = kde_x.size
    kde_T = kde_x[-1] - kde_x[0] # Duration over which kernel density was estimated
    fs = (kde_x.size/(kde_T)) # Sampling frequency of KDE

    # Determine dominant frequency using FFT
    freq_axis = np.arange(N)/kde_T
    kde_fft = np.abs(np.fft.fft(kde_y))
    fft_maxima = find_peaks(kde_fft)[0] # Intermediate variable; includes aliases
    fft_maxima = fft_maxima[fft_maxima < N/2] # Exclude maxima greater than Nyquist frequency
    dominant_frequency_contenders = freq_axis[fft_maxima] # Frequencies where maxima occured
    dominant_freq_fft = dominant_frequency_contenders[np.argmax(kde_fft[fft_maxima])] # The frequency corresponding to gloabl maximum

    # Determine dominant frequency in time domain
    kde_y_maxima = find_peaks(kde_y, height=0.05*np.amax(kde_y))[0]
    if kde_y_maxima.size > 1:
        dominant_freq_time = len(kde_y_maxima)/kde_T
    else:
        dominant_freq_time = np.NAN
        dominant_freq_fft = np.NAN

    # Visualize time and frequency domains
    if plot == True:
        dominant_frequency_plot(
            kde_x,
            kde_y,
            freq_axis,
            kde_fft,
            kde_y_maxima,
            fft_maxima
        )
    return dominant_freq_fft, dominant_freq_time
    
def average_theta_freq(theta_x: np.ndarray, theta_y: np.ndarray) -> float:
   
    """Compute the average theta oscillation frequency on some interval.

    Args:
        theta_x: time axis of theta oscillation in seconds.
        theta_y: amplitude time series of theta oscillation.
    Returns:
        theta_freq: Frequency of theta oscillations on given interval; np.NAN if 
            no maxima occurred on interval.

    """

    theta_T = theta_x[-1] - theta_x[0]
    theta_y_maxima = find_peaks(theta_y, height=0.05*np.amax(theta_y))[0]
    if theta_y_maxima.size > 1:
        theta_freq = len(theta_y_maxima)/theta_T
    else:
        theta_freq = np.NAN
    return theta_freq

def prep_pyr_qpspike(qpspike: np.ndarray, intspike: np.ndarray, pyrspike: np.ndarray) -> tuple:

    """Ensure spike-aligned data is organized appropriately and extract precursor constructs.

    Args:
        qpspike: (sum(num_trials[pyr]), 3) array, with trial index in first column, spike times in 
            second column and neuron label in third column. Trials were all time-shifted to satisfy 
            maximum likelihood estimation for a single pyr neuron. 
        intspike: (sum(num_spikes[int]), 2) array, with interneuron label in first column and spike
            times in second column.
        pyrspike: (sum(num_spikes[pyr]), 2) array, with pyramidal neuron label in first column and 
            spike times in second column.
    Raises:
        Exception: Whenever pyramidal neuron labels or trial labels are not in ascending order.
    Returns:
        pyr_com_trials: list of np.ndarrays, where each array contains all spike times across trials
            for each neuron. Indexes return neuron labels in pyr_com_neuron_labels.  
        pyr_sep_trials: list of np.ndarrays, where each array contains spike times for individual
            trials. Indexes return neuron and trial labels in pyr_sep_neuron_labels and 
            pyr_sep_trial_labels respectively.
        pyr_com_neuron_labels: maps indices in pyr_com_trials to the correct neuron label.
        pyr_sep_neuron_labels: maps indices in pyr_sep_trials to the correct neuron label.
        pyr_sep_trial_labels: maps indices in pyr_sep_trials to the correct trial label.

    """

    # Remove interneuron data from qpspike
    interneuron_indices = []
    for i, label in enumerate(qpspike[:, 2]):
        if label in intspike[:, 0]:
            interneuron_indices.append(i)
    pyr_qpspike = np.delete(qpspike, interneuron_indices, axis=0) 

    # Notify on missing pyramidal data
    missing_pyr = []
    all_pyr_labels = np.array(list(set(pyrspike[:, 0])))
    pyr_com_neuron_labels = np.array(list(set(pyr_qpspike[:, 2])))
    for label in all_pyr_labels:
        if label not in pyr_com_neuron_labels:
            missing_pyr.append(label)
    print(f'Pyramidal neurons {missing_pyr} absent from qpspike.')

    # Ensure qpspike data is sorted appropriately and construct relevant label vectors
    if not all(np.diff(pyr_qpspike[:, 2]) >= 0): # Ensure neuron labels are in ascending order
        raise Exception('qpspike pyramidal neuron labels are not in ascending order.')
    
    # Iterate over each unique pyramidal label appearing in qpspike
    pyr_num_trials = [] # Used to construct trial-separated label maps 
    pyr_com_trials = [] # Contains all spike times for one neuron across trials
    for label in pyr_com_neuron_labels: 
        
        # Extract all rows with corresponding label
        neuron_data = pyr_qpspike[np.where(pyr_qpspike[:, 2] == label)] 

        # Ensure trials are ascending
        trial_FOD = np.diff(neuron_data[:, 0]) # Intermediate parameter: trial label first order derivative
        if not np.all(trial_FOD >= 0):
            raise Exception(f'qpspike trials are not in ascending order for neuron: {label}')

        pyr_num_trials.append(len(set(neuron_data[:, 0]))) 
        pyr_com_trials.append(np.sort(neuron_data[:, 1])) 

    # Create vector to map index in trial-separated data to neuron label
    pyr_sep_neuron_labels = [] 
    for label_i, num_trials in enumerate(pyr_num_trials):
        pyr_sep_neuron_labels.extend([pyr_com_neuron_labels[label_i] for k in range(num_trials)])
    
    # Create vector to map index in trial-separated data to trial label
    trial_labels = pyr_qpspike[:, 0]
    selection = np.ones(trial_labels.shape[0], dtype=bool)
    selection[1:] = trial_labels[1:] != trial_labels[:-1]
    pyr_sep_trial_labels = trial_labels[selection]

    # Create vector of trial-separated data
    trial_partition_indices = partition_spikes(pyr_qpspike)
    pyr_sep_trials = separate_neuron_data(trial_partition_indices, pyr_qpspike)

    return pyr_com_trials, pyr_sep_trials, pyr_com_neuron_labels, pyr_sep_neuron_labels, pyr_sep_trial_labels

def shift_qspike(unshifted_qspike:np.ndarray, trial_shifts: np.ndarray) -> np.ndarray:

    """Shift qspike to create qpspike satisfying maximum likelihood estimate.

    Args:
        unshifted_qspike: first column indicates trial number, second is spike times (s), third is
            neuron label.
        trial_shifts: vector of time shifts (in s) to maximize likelihood estimate for spike times
            over a 4 second trial window -- index corresponds to trial number.
    Returns:
        qpspike: Data structure is identical to unshifted qspike, but with spike time values
            corrected. Only spike remaining on the original [0, 4] interval are kept.
    """

    qpspike = unshifted_qspike

    for trial, shift in enumerate(trial_shifts, start=1):

        trial_rows = np.where(unshifted_qspike[:, 0] == trial)[0]
        trial_coords = (trial_rows, np.ones((trial_rows.shape[0],), dtype=int))
        qpspike[trial_coords] = qpspike[trial_coords] + shift
    
    return qpspike

def dominant_frequency_analysis(
    kde_y: np.ndarray,
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    window_dur: float,
    sampling_freq: int
    ) -> float:

    """Find maximum
    """

    # Define constants and determine if there is some post-window remainder
    window_N = int(sampling_freq*window_dur)
    num_windows = theta_x.size - window_N 
    kde_threshold = 0.05*np.amax(kde_y)
    theta_threshold = 0.05*np.amax(theta_y)
    
    window_detuning = np.empty((num_windows,), dtype=float)

    # Compute kde-theta detuning over each complete window  
    for k in range(num_windows):

        kde_y_maxima = find_peaks(kde_y[k:window_N + k], height=kde_threshold)[0]

        # If kde frequency is nonzero, compute it and proceed to compute theta frequency in the same manner 
        if kde_y_maxima.size > 1:
            kde_freq = len(kde_y_maxima)/window_dur

            theta_y_maxima = find_peaks(theta_y[k:window_N + k], height=theta_threshold)[0]
            if theta_y_maxima.size > 1:
                theta_freq = len(theta_y_maxima)/window_dur
            else:
                theta_freq = np.NAN   

        # Otherwise don't bother computing theta frequency
        else:
            kde_freq = np.NAN
            theta_freq = np.NAN

        window_detuning[k] = kde_freq - theta_freq # Compute detuning over window
    
    if np.any(~np.isnan(window_detuning)):
        max_detuning = np.amax(window_detuning[~np.isnan(window_detuning)])
    else:
        max_detuning = np.NAN

    return max_detuning

def compute_PRQ(
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    spike_times: np.ndarray,
    theta_fs: float,
    discont=0.95,
    check_analytic=False
    ) -> float:

    """Compute Phase Relationship Quantification (PRQ).

    Args:
        theta_x: Time axis over which values in theta_y correspond.
        theta_y: Theta amplitude in time domain.
        spike_times: Relative spike times (s) -- where t0=0 is the start of a trial.
        theta_fs: Sampling frequency of theta oscillation
        discont: Magnitude of discontinuities in wrapped signal phase (discont*2*pi)
            required to define a cycle boundary; 0 < discont < 1.
        check_analytic: For debugging -- whether or not to visualize analytic signal
            and the cycle boundaries in theta_y. 
    Returns:
        prq: Signed quantity indicating apparent phase relationship with magnitude
            possibly characterizing some notion of relationship 'strength'. 

    """

    # Convert spike times to indices
    spike_indices = (spike_times*theta_fs).astype(int)

    # Create analytic signal
    analytic_theta = hilbert(theta_y)

    # Prepare LFP phase and detect cycle boundaries (where analytic signal crosses positive x axis) 
    theta_phase = np.arctan2(analytic_theta.imag, analytic_theta.real) # Phase from analytic signal, wrapped [-pi, pi]
    theta_phase[theta_phase < 0] += 2*np.pi # Shift from interval [-pi, pi] to [0, 2pi]
    theta_omega = np.diff(theta_phase) # Instantaneous frequency in wrapped phase 
    cycle_boundary_i = np.where(theta_omega < -discont*2*np.pi)[0] # Indices where phase resets
    CT_times = []

    # Plot the analytic signal and cycle boundaries derived from it if requested 
    if check_analytic == True:
        analytic_boundaries_plot(
            theta_x,
            theta_y,
            cycle_boundary_i,
            analytic_theta
        )

    # Unwrap the phase to prevent reset effects
    theta_phase = np.unwrap(theta_phase, discont=None, period=2*np.pi)

    # Approximate spike phase central tendency (CT) on each cycle -- 0 indicates undefined
    cycle_phi_CT = np.zeros((cycle_boundary_i.shape[0] + 1,)) # For n boundaries, there are n+1 cycles

    # Iterate over each boundary 
    for i, boundary_i in enumerate(cycle_boundary_i):

        # First cycle includes all spikes on [:boundary]
        if i == 0:
            cycle_spike_indices = spike_indices[spike_indices < boundary_i]
            cycle_phi = theta_phase[cycle_spike_indices]
            if cycle_phi.size != 0: # If CT is defined
                cycle_phi_CT[i] = np.mean(cycle_phi)
                CT_times.append(np.mean(theta_x[cycle_spike_indices]))
        
        # Subsequent cycles include spikes on [boundary_{i-1}:boundary_{i}]
        else:
            cycle_spike_indices = spike_indices[(spike_indices < boundary_i) & (spike_indices >= cycle_boundary_i[i - 1])]
            cycle_phi = theta_phase[cycle_spike_indices]
            if cycle_phi.size != 0: # If CT is defined
                cycle_phi_CT[i] = np.mean(cycle_phi)
                CT_times.append(np.mean(theta_x[cycle_spike_indices]))
    
    # The final cycle includes spikes on [boundary_{n}:]
    cycle_spike_indices = spike_indices[spike_indices > boundary_i]
    cycle_phi = theta_phase[cycle_spike_indices]
    if cycle_phi.size != 0: # If CT is defined
        cycle_phi_CT[-1] = np.mean(cycle_phi)
        CT_times.append(np.mean(theta_x[cycle_spike_indices]))
    
    # Mask undefined central tendencies
    cycle_phi_CT = np.ma.masked_equal(cycle_phi_CT, 0)

    # Compute backward first order difference between consecutively defined cycles
    cycle_phi_CT_back_FOD = np.diff(np.flip(cycle_phi_CT))

    # Extract only the defined differences -- those that occur between consecutive cycles
    cycle_phi_CT_back_FOD = cycle_phi_CT_back_FOD.compress(~np.isnan(cycle_phi_CT_back_FOD))

    # Phase relationship quantification is the CT of the defined backward differences
    prq = np.mean(cycle_phi_CT_back_FOD)
    
    return prq, CT_times
    


    
        





