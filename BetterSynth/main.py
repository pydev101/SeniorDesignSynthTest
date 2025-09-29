import time
import numpy as np
import sounddevice as sd
import threading
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


# Parameters
sample_rate = 44100  # samples per second
block_size = 1024    # block size

# For FFT plotting
fft_data = np.zeros(block_size)
freqs = fftfreq(block_size, 1 / sample_rate)

fig, ax = plt.subplots()
line, = ax.plot(freqs[:block_size // 2], np.zeros(block_size // 2))
ax.set_xlim(0, 5000)  # Show up to 5 kHz
ax.set_ylim(0, 1)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude")
plt.ion()

phase = 0

def audio_callback(outdata, frames, time, status):
    global phase, fft_data

    t = (np.arange(frames) + phase) / sample_rate
    wave = np.zeros_like(t)

    tStart = phase / sample_rate # Start time of frame in s
    print(tStart)

    wave += np.sin(2 * np.pi * 300 * t)*np.exp(-t/1)
    wave += np.sin(2 * np.pi * 600 * t)*np.exp(-t/2)

    outdata[:] = wave.reshape(-1, 1)

    # Save block for FFT
    fft_data = wave.copy()
    phase += frames

def update_plot():
    global fft_data
    Y = fft(fft_data)
    Y = np.abs(Y[:block_size // 2]) / (block_size / 2)

    line.set_ydata(Y)
    ax.set_ylim(0, max(1, np.max(Y)))
    plt.pause(0.001)

with sd.OutputStream(channels=1, callback=audio_callback, samplerate=sample_rate, blocksize=block_size):
    while True:
        update_plot()
        sd.sleep(1)
