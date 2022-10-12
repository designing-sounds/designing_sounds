import numpy as np


class SinWave:
    def __init__(self):
        self._freq = 1
        self._amp = 1
        self._phase = 0

    def get_array(self, samples):
        x = np.linspace(0, 2 * np.pi, samples)
        return self._amp * np.sin(x * 2 * np.pi * self._freq) + self._phase

    def set_amp(self, amp):
        self._amp = amp

    def set_freq(self, freq):
        self._freq = freq

    def set_phase(self, phase):
        self._phase = phase
