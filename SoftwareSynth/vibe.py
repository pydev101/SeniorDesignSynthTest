import time

import numpy as np
import sounddevice as sd
import threading
import math

# Parameters
sample_rate = 44100  # samples per second
block_size = 1024    # how many samples to process per block

noteLock = threading.Lock()
phase = 0

# freq, duration, startCycle
notesPlaying = []

def playNote(f, dur):
    with noteLock:
        notesPlaying.append([f, dur, None])

def audio_callback(outdata, frames, time, status):
    global phase
    global notesPlaying

    t = (np.arange(frames) + phase) / sample_rate # Generate time values for the new 1024 frames taking into account the previous time the other frames took

    wave = np.zeros_like(t)

    with noteLock:
        if len(notesPlaying) > 0:
            notesToRemove = []
            for i, v in enumerate(notesPlaying):
                if v[2] is None:
                    v[2] = phase # Set start frame (time) as 3rd argument

                frameLengthOfNote = v[1] * sample_rate

                print(frameLengthOfNote - (phase - v[2]))
                if frameLengthOfNote - (phase - v[2]) < block_size:
                    # Do something special for the last block
                    # TODO Play last segment of last note
                    print("L")
                    notesToRemove.append(i)
                else:
                    # Sound lasts for more than a frame
                    wave += 0.5 * np.sin(2 * np.pi * v[0] * t)
            notesToRemove.reverse()
            for note in notesToRemove:
                del notesPlaying[note]

    # Write to output buffer (stereo if desired)
    outdata[:] = wave.reshape(-1, 1)

    # Update phase so waveform is continuous
    phase += frames

startTime = time.time()
realF = 1000
# Open a stream
with sd.OutputStream(channels=1, callback=audio_callback, samplerate=sample_rate, blocksize=block_size):
    while True:
        dT = time.time() - startTime
        if dT > 1:
            playNote(realF, 0.5)
            realF += 500
            startTime = time.time()

        sd.sleep(1)