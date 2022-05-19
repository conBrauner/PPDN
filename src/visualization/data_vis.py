# Standard library imports
import warnings

# Third party imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def raster_plot(spike_times: list) -> None:
   
    """Create and show a raster plot depicting spike times.

    Args:
        spike_times: 2D list; each sublist pertains to a separate neuron.

    """

    plt.close()
    fig, ax = plt.subplots()

    fig.patch.set_alpha(0.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.eventplot(
        spike_times,
        colors='k',
        lineoffsets=1.1
        )
    plt.show()

def locomotion_plot(position: np.ndarray, T: float) -> None: 
    
    """Plot 2D animal position and color to indicate passage of time.

    Args:
        position: x and y coordinates in array of shape (2, num_timesteps).
        T: Total duration (in seconds) over which locomotion transpired.

    """

    plt.close()
    fig, ax = plt.subplots()

    fig.patch.set_alpha(0.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    color_scalar = np.linspace(0, 1, num=position.shape[1])
    ax.scatter(
        position[0, :],
        position[1, :],
        c=color_scalar,
        s=0.1,
        cmap='viridis'
        )

    colorbar_labels = list(np.linspace(0, T, num=6, dtype=int).astype(str))
    colorbar=plt.colorbar(cm.ScalarMappable(norm=None, cmap='viridis'), ax=ax)
    colorbar.ax.set_yticklabels(colorbar_labels)
    colorbar.set_label('Time $(s)$', rotation=270)
    plt.show()

def dominant_frequency_plot(
    kde_x: np.ndarray,
    kde_y: np.ndarray,
    freq_axis: np.ndarray,
    kde_fft: np.ndarray,
    kde_y_maxima: np.ndarray,
    kde_fft_maxima: np.ndarray
    ) -> None:

    """Plot spike time kde in time and frequency domain and label maxima.

    Args:
        kde_x: The domain over which KDE was computed.
        kde_y: The probability density estimate over kde_x.
        freq_axis: Valid frequency domain for discrete FFT.
        kde_fft: Frequency content of kde_y over freq_axis.
        kde_y_maxima: Peak indices of kde_y.
        kde_fft_maxima: Peak indices of kde_fft.
    
    """

    fig, axes = plt.subplots(2, 1)
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].plot(kde_x, kde_y, linewidth=0.8, color='k')
    axes[0].scatter(kde_x[kde_y_maxima], kde_y[kde_y_maxima], s=10, color='r')
    axes[1].plot(freq_axis, kde_fft)
    axes[1].scatter(freq_axis[kde_fft_maxima], kde_fft[kde_fft_maxima], s=10, color='r')

    plt.show()

def spike_frequency_histogram(frequencies: np.ndarray, num_bins: int) -> None:
    """Plot histogram of spike frequency for each neuron.

    Args:
        frequencies: spike frequencies in Hz.
        num_bins: integer value for number of data bins.

    """

    plt.close()
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.hist(frequencies, num_bins, color='k', rwidth=0.8, density=True)

    plt.show()

def PR_plot(kde_y: np.ndarray, theta_x: np.ndarray, theta_y: np.ndarray) -> None:
    
    """Plot spike kde and theta oscillation to visualize phase relationship (PR).

    Args:
        kde_y: The probability density estimate over theta_x.
        theta_x: The segment of time axis corresponding to kde_x.
        theta_y: Theta oscillation amplitude over theta_x.
    
    """

    plt.close()
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[0].plot(theta_x, kde_y, linewidth=0.8, color='k')
    axes[1].plot(theta_x, theta_y, linewidth=0.8, color='k')

    plt.show()

def analytic_boundaries_plot(
    signal_x: np.ndarray,
    signal_y: np.ndarray,
    boundaries: np.ndarray,
    analytic_signal: np.ndarray
    ) -> None:

    """Plot analytic signal in complex plane alongside original signal with cycle boundaries.
    
    Args: 
        signal_x: Time axis over which values in signal_y correspond.
        signal_y: Signal amplitude in time domain.
        boundaries: Indices in signal corresponding to large discontinuities in analytic_signal.
        analytic_signal: For discrete signal x(n), analytic_signal is A(n) = x(n) + iH(x(n)),
            where H(f) is the discrete Hilbert transform of signal f.

    """

    # Figure boilerplate
    plt.close()
    fig, axes = plt.subplots(1, 2)
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Plot analytic signal as a scatterplot with a colormapping to indicate time
    color_scalar = np.linspace(0, 1, num=signal_x.shape[0])
    axes[0].scatter(
        analytic_signal.real,
        analytic_signal.imag,
        c=color_scalar,
        s=0.1,
        cmap='viridis'
        )
    
    # Plot the signal and cycle boundaries identified in analytic signal 
    axes[1].plot(signal_x, signal_y, linewidth=0.8, color='k')
    axes[1].vlines(
        signal_x[boundaries],
        np.amin(signal_y),
        np.amax(signal_y),
        color='r',
        linestyle='--'
        )
    
    # Customize labels of colorbar
    colorbar_labels = list(
        np.linspace(
            np.amin(signal_x),
            np.amax(signal_x),
            num=6,
            dtype=int
            )
        )
    
    # Create colorbar and add it to the ax object containing analytic signal
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        colorbar = plt.colorbar(cm.ScalarMappable(norm=None, cmap='viridis'), ax=axes[0])
        colorbar.ax.set_yticklabels(colorbar_labels)
        colorbar.set_label('Time $(s)$', rotation=270)

    plt.show()
    
