import numpy as np
import pyaudio
from kivy.core.window import Window
from src.wave_model.wave_model import SoundModel


class WaveSound:
    def __init__(self, sample_rate: int, waveform_duration: float, chunk_duration: float, sound_model: SoundModel):
        self.waveform_duration = waveform_duration
        self._chunk_index = 0
        self.chunk_duration = chunk_duration
        self.sound_model = sound_model
        self._is_playing = False
        self.sample_rate = sample_rate
        self._py_audio = pyaudio.PyAudio()
        self._stream = self._py_audio.open(format=pyaudio.paFloat32, channels=1, rate=self.sample_rate, output=True,
                                           stream_callback=self.callback,
                                           frames_per_buffer=int(self.sample_rate * self.chunk_duration))
        self._stream.stop_stream()
        Window.bind(on_request_close=self.shutdown_audio)

    def callback(self, _in_data, _frame_count, _time_info, _flag):
        sound: np.ndarray = self.sound_model.model_sound(self.sample_rate, self.chunk_duration,
                                                         start_time=self._chunk_index * self.chunk_duration)
        self._chunk_index = self._chunk_index + 1
        return sound, pyaudio.paContinue

    def is_playing(self) -> bool:
        return self._is_playing

    def play_audio(self):
        self._is_playing = True
        self._stream.start_stream()

    def pause_audio(self):
        self._is_playing = False
        self._stream.stop_stream()

    def sound_changed(self):
        self._chunk_index = 0

    def shutdown_audio(self, _) -> bool:
        self._stream.close()
        self._py_audio.terminate()
        return False
