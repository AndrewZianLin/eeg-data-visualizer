# === Import statements ===

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import threading





# === Functions ===

# Read new data asynchronously in background thread
def data_acquisition_and_update():
    while True:
        # --- Read new data ---

        # Block until the main thread requests a new data read
        update_event.wait() # Wait for the event to trigger a data read
        
        # Read the last *num_samples_updated* samples
        new_data = stream.read(num_samples_updated)
        new_data_array = np.frombuffer(new_data, dtype=np.int16)



        # --- Update stream graph ---

        # Update the buffer with the latest data
        data_buffer[:-num_samples_updated] = data_buffer[num_samples_updated:] # Shift previous data left
        data_buffer[-num_samples_updated:] = new_data_array # Add the latest data to the end of the buffer

        # Update stream graph with the new data
        stream_line.set_ydata(data_buffer)



        # --- Update frequency graph ---

        # Zero-pad the FFT sample for more efficient FFT computation
        # Just use *fft_sample* for .fft() and *fft_sample_size* for .fftfreq() if memory strain is too high
        fft_sample = data_buffer[-fft_sample_size:]
        padded_fft_sample = np.zeros(padded_fft_sample_size, dtype=np.int16)
        padded_fft_sample[:fft_sample_size] = fft_sample
        
        # Get the frequencies of *fft_sample*
        fft_result = np.fft.fft(padded_fft_sample)
        fft_bins = np.fft.fftfreq(padded_fft_sample_size, 1/RATE)

        # Only use frequencies up to Nyquist limit (positive frequencies)
        positive_fft_result = fft_result[:len(fft_result)//2]
        positive_fft_bins = fft_bins[:len(fft_bins)//2]
        
        # Find the magnitudes of each brainwave type for *fft_sample*
        for i in range(0, len(brainwave_types)):
            # Get frequency range for the current brain wave type
            freq_range = list(brainwave_types.values())[i][0]

            # List of indices for frequency bins within the current frequency range
            range_bins = np.where((positive_fft_bins >= freq_range[0]) & (positive_fft_bins <= freq_range[1]))

            # Sum the magnitudes of the frequencies within this range
            total_range_magnitude = np.sum(np.abs(positive_fft_result[range_bins]))

            # Update the corresponding bar in the frequency graph with the new magnitude
            freq_bars[i].set_height(total_range_magnitude)

        # Autoscale the frequency graph according to the new magnitudes
        freq_graph.relim()
        freq_graph.autoscale_view()
        
        

# Update stream and frequency graphs
def update_graphs(frame):
    update_event.set() # Signal the data acquisition thread to read new data
    return stream_line, *freq_bars # Return updated graphs





# === PyAudio setup ===

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open data stream
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
)





# === Figure setup ===

# Figure setup
figure, axis = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
figure.tight_layout(pad=5)
figure.canvas.manager.set_window_title("EEG data visualizer")

# Settings
seconds = 1.0 # Display *seconds* worth of samples in the graph
num_samples_displayed = int(RATE * seconds)
update_interval = 20 # Update the graph every *update_interval* milliseconds
num_samples_updated = int(RATE * update_interval / 1000) # Update graph with *update_interval* worth of samples
amplitude_limit = 2000 # Maximum amplitude that can be displayed on the stream graph; 16 bit maximum is 2**15-1
fft_sample_ratio = 0.50 # Ratio of *data_buffer* to use in the FFT
                        # Smaller ratio = faster response but less frequency resolution
                        # *RATE* / *padded_fft_sample_size* MUST >= 4 to capture all brain wave types
                        # Alternatively, *RATE* / *fft_sample_size* must >= 4 to capture all brain wave types if zero-padding has been disabled to lower memory strain
fft_sample_size = int(num_samples_displayed * fft_sample_ratio)
padded_fft_sample_size = int(math.pow(2, math.ceil(math.log2(fft_sample_size)))) # Next highest power of 2 from *fft_sample_size*, used for zero-padding





# === Stream graph setup ===

# Initialize a buffer for the last *seconds* worth of samples
data_buffer = np.zeros(num_samples_displayed, dtype=np.int16)

# Initialize stream graph
stream_graph = axis[0]
stream_graph.set_title("EEG data stream")
stream_graph.set_xlabel("Time (s)")
stream_graph.set_ylabel("Amplitude")
stream_graph.set_xlim(-seconds, 0)
stream_graph.set_ylim(-amplitude_limit, amplitude_limit)

x = np.linspace(-seconds, 0, num_samples_displayed)
y = data_buffer
stream_line, = stream_graph.plot(x, y)





# === Frequency graph setup ===

# Brainwave freqency ranges and bar colors
brainwave_types = {
    "Delta": [(0.5, 4), "red"],
    "Theta": [(4, 8), "darkorange"],
    "Alpha": [(8, 13), "gold"],
    "Beta": [(13, 32), "limegreen"],
    "Gamma": [(32, 100), "dodgerblue"]
}

# Initialize frequency graph
freq_graph = axis[1]
freq_graph.set_title("Brain wave relative magnitudes")
freq_graph.set_xlabel("Brain wave type")
freq_graph.set_ylabel("Relative magnitude")
freq_graph.set_yticks([])
freq_graph.set_yticklabels([])

freq_bars = freq_graph.bar(
    list(brainwave_types.keys()),
    np.zeros(len(brainwave_types)),
    color=[wave_type[1] for wave_type in list(brainwave_types.values())],
    alpha=0.7
)





# === Data acquisition and update thread setup ===

update_event = threading.Event()  # Event to trigger data reading and graph updating
acquisition_and_update_thread = threading.Thread(target=data_acquisition_and_update, daemon=True)
acquisition_and_update_thread.start() # Start the background thread





# === Show graphs ===

# Update the graphs in real time
ani = FuncAnimation(
    figure,
    update_graphs,
    blit=False,
    cache_frame_data=False,
    interval=update_interval
)

# Show graphs
plt.show()

# Close the data stream and terminate PyAudio after the graph is closed
stream.stop_stream()
stream.close()
p.terminate()