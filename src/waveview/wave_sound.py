import numpy as np
import pyaudio
from kivy.core.window import Window
from src.wave_model.wave_model import SoundModel


class WaveSound:
    def __init__(self, sample_rate: int, sound_duration: float, chunk_duration: float, sound_model: SoundModel):
        self.sound_duration = sound_duration
        self.chunk_index = 0
        self.chunk_duration = chunk_duration
        self.sound_model = sound_model
        self.is_playing = False
        self.sample_rate = sample_rate
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.sample_rate, output=True,
                                  stream_callback=self.callback, frames_per_buffer=int(self.sample_rate * self.chunk_duration))
        self.stream.stop_stream()
        Window.bind(on_request_close=self.shutdown_audio)

    def callback(self, in_data, frame_count, time_info, flag):
        sound: np.ndarray = self.sound_model.model_sound(self.sample_rate, self.chunk_duration, self.chunk_index * self.chunk_duration)
        self.chunk_index = (self.chunk_index + 1) % (self.sound_duration / self.chunk_duration)
        return sound, pyaudio.paContinue

    def press_button_play(self) -> None:
        if not self.is_playing:
            self.is_playing = True
            self.stream.start_stream()
        else:
            self.is_playing = False
            self.stream.stop_stream()

    def shutdown_audio(self, *args) -> bool:
        self.stream.close()
        self.p.terminate()
        return False
