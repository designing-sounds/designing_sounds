import numpy as np


class SinWave:
    def __init__(self, freq: float, amp: float):
        self.freq = freq
        self.amp = amp

    def get_array(self, samples: int) -> np.array:
        x = np.linspace(0, 2 * np.pi, samples)
        return self.amp * np.sin(x * self.freq)
