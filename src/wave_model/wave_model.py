import numpy as np


class SinWave:
    def __init__(self, freq: float, amp: float):
        self.freq = freq
        self.amp = amp

    def get_array(self, sample_rate: int, duration: float = 1) -> np.array:
        points = np.linspace(0, duration, int(duration * sample_rate), endpoint=False, dtype=np.float32)
        return self.amp * np.sin(self.freq * points * 2 * np.pi)
