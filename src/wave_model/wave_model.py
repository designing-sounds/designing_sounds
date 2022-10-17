import numpy as np

MAX_SOUND_VAL = 32767


def normalize_sound(arr: np.ndarray) -> np.ndarray:
    sound = arr * MAX_SOUND_VAL
    return sound.astype(np.int16)


class SinWave:
    def __init__(self, freq: float, amp: float, time: float):
        self.freq = freq
        self.amp = amp
        self.time = time

    def get_array(self, sample_rate: int) -> np.array:
        points = np.linspace(0, self.time, int(self.time * sample_rate))
        return self.amp * np.sin(self.freq * points * 2 * np.pi)
