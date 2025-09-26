import numpy as np
import pyaudio
import time
import threading
from scipy.ndimage import shift
import sounddevice as sd

fs = 44000 # Sample frequency Hz, contant
duration = 10. # second, arbitrary length of entire sound

buffer = np.zeros(int(duration*fs), dtype = np.int16)
buffMax = 2**16
buffMin = 0
buffLock = threading.Lock()

def genSine(f, dur):
    timeArr = np.arange(0, dur, 1/fs)
    tempBuff = np.zeros(len(buffer), dtype=np.int16)

    tempBuff[0:len(timeArr)] = np.sin(2*np.pi*f*timeArr)*32768
    return tempBuff

P = pyaudio.PyAudio()
stream = P.open(rate=fs, format=pyaudio.paInt16, channels=1, output=True)

def playBuffer():
    global buffer
    global buffLock
    N = int(0.5*fs)

    i = 0
    while i<len(buffer):
        with buffLock:
            tempBuff = buffer[0:N].copy()
            buffer = shift(buffer, -N, cval=0)
            tempBuff = np.clip(tempBuff, buffMin, buffMax)
        stream.write(tempBuff.tobytes())


buffer = buffer + genSine(500, 4) + genSine(300, 2) + + genSine(1000, 6)
playBuffer()


stream.close() # this blocks until sound finishes playing

P.terminate()

# Sources
# https://www.scivision.dev/playing-sounds-from-numpy-arrays-in-python/


# import sounddevice as sd
# sd.play(myarray, 44100)