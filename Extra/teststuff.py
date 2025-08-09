import numpy as np
import matplotlib.pyplot as plt

# Parameters
duration = 4  # seconds
sampling_rate = 256  # Hz
frequency = 5  # Hz, typical alpha wave frequency
amplitude = 100  # μV
phase = 0

# Time vector
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
# Cosine signal
cosine = 100 * np.cos(2 * np.pi * frequency * t + phase)

#sine signal
sine = 18 * np.sin(2 * np.pi * frequency * t + phase)

linear = amplitude * (frequency * t + phase)

# Visualize
plt.plot(t, cosine)
plt.plot(t, sine)
plt.plot(t, linear)
plt.title("Synthetic Cosine Wave")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [μV]")
plt.grid()
plt.show()
