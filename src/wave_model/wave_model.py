import numpy as np


def normalize_sound(arr: np.ndarray) -> np.ndarray:
    sound = arr
    return sound.astype(np.float32)


class SinWave:
    def __init__(self, freq: float, amp: float):
        self.freq = freq
        self.amp = amp

    def get_array(self, sample_rate: int, duration: float = 1) -> np.array:
        points = np.linspace(0, duration, int(duration * sample_rate))
        return self.amp * np.sin(self.freq * points * 2 * np.pi)
