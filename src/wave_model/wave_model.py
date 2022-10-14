import numpy as np

MAX_SOUND_VAL = 32767


def normalize_sound(arr: np.ndarray) -> np.ndarray:
    sound = arr * MAX_SOUND_VAL
    return sound.astype(np.int16)


class SinWave:
    def __init__(self, freq: float, amp: float):
        self.freq = freq
        self.amp = amp

    def get_array(self, samples: int) -> np.array:
        x = np.linspace(0, 2 * np.pi, samples)
        return self.amp * np.sin(x * self.freq)
