import time
import numpy as np
import sounddevice as sd
import threading
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import keyboard

# Parameters
sample_rate = 44100  # samples per second
block_size = 1024  # block size

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


class Note:
    # Attack (s)
    # Decay (s)
    # Sustain (amplitude)
    # Release (s)
    def __init__(self, freq, attack, decay, sustain, release):
        self.freq = freq
        self.A = attack
        self.D = decay
        self.S = sustain
        self.R = release

        self.amplitude = 0
        self.startTime = None

        self.envelope = np.vectorize(self.envelopeFunct)

    def getClip(self, t):
        if self.startTime is None:
            self.startTime = t[0]
        t = t - self.startTime

        self.amplitude = self.envelope(t)
        return np.multiply(self.amplitude, np.sin(2 * np.pi * self.freq * t))

    def envelopeFunct(self, t):
        aMax = 1
        buttonDown = True

        if buttonDown:
            amp = t * (aMax/self.A) if t < self.A else 0
            amp = aMax - ((t-self.A) * ((aMax-self.S) / self.D)) if self.A < t < self.A+self.D else amp
            amp = self.S if self.A+self.D < t else amp
        else:
            pass

        # amp = aMax - ((self.D - self.A) * (1 / self.D)) if self.A+self.D < t < self.A + self.D + self.S else amp
        if amp < 0:
            return 0
        return amp

notes = [Note(300, 1, 1, 0.5, 1), Note(432, 0.3, 8, 0.5, 1)]

def audio_callback(outdata, frames, time, status):
    global phase, fft_data

    t = (np.arange(frames) + phase) / sample_rate
    wave = np.zeros_like(t)

    tStart = phase / sample_rate  # Start time of frame in s


    for n in notes:
        wave += n.getClip(t)
    # if t[0] > 1.2:
        # wave += notes[1].getClip(t)
    # wave += np.sin(2 * np.pi * 300 * t) * np.exp(-t / 1)
    # wave += np.sin(2 * np.pi * 600 * t) * np.exp(-t / 2)

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
