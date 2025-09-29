import time
import numpy as np
import sounddevice as sd
import threading
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from pynput import keyboard

current_keys = set()

def on_press(key):
    try:
        current_keys.add(key.char)  # Normal keys
    except AttributeError:
        current_keys.add(str(key))  # Special keys like Key.space
    print("Pressed:", current_keys)

def on_release(key):
    try:
        current_keys.discard(key.char)
    except AttributeError:
        current_keys.discard(str(key))
    print("Released:", current_keys)

    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Parameters
sample_rate = 44100  # samples per second
block_size = 1024    # block size

noteLock = threading.Lock()
phase = 0
notesPlaying = []

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

def playNote(f, dur):
    with noteLock:
        notesPlaying.append([f, dur, None])

def audio_callback(outdata, frames, time, status):
    global phase, notesPlaying, fft_data

    t = (np.arange(frames) + phase) / sample_rate
    wave = np.zeros_like(t)

    with noteLock:
        if len(notesPlaying) > 0:
            notesToRemove = []
            for i, v in enumerate(notesPlaying):
                if v[2] is None:
                    v[2] = phase

                frameLengthOfNote = v[1] * sample_rate
                if frameLengthOfNote - (phase - v[2]) < block_size:
                    notesToRemove.append(i)
                    # Do something special for the last block
                    # TODO Play last segment of last note
                    # print("L")
                else:
                    wave += 0.5 * np.sin(2 * np.pi * v[0] * t)
            for note in reversed(notesToRemove):
                del notesPlaying[note]

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

startTime = time.time()
realF = 1000

with sd.OutputStream(channels=1, callback=audio_callback, samplerate=sample_rate, blocksize=block_size):
    while realF <= 4500:
        dT = time.time() - startTime
        if dT > 1:
            # playNote(realF, 0.5)
            realF += 500
            startTime = time.time()
        if dT > 0.5:
            playNote(600+5*np.sin(2*3.14*time.time()/0.1), 0.5)
            # playNote(400 + 300 * np.sin(2 * 3.14 * time.time() / 1.5), 0.5)

        update_plot()
        sd.sleep(1)
