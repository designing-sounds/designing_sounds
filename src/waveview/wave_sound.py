import numpy as np
import pyaudio
from kivy.core.window import Window


class WaveSound:
    def __init__(self, sample_rate, sound_duration, chunk_time, model):
        self.sound_duration = sound_duration
        self.chunk_index = 0
        self.chunk_time = chunk_time
        self.model = model
        self.is_playing = False
        self.sample_rate = sample_rate
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.sample_rate, output=True,
                                  stream_callback=self.callback, frames_per_buffer=int(self.sample_rate * self.chunk_time))
        self.stream.stop_stream()
        Window.bind(on_request_close=self.shutdown_audio)

    def callback(self, in_data, frame_count, time_info, flag):
        sound: np.ndarray = self.model.model_sound(self.sample_rate, self.chunk_time, self.chunk_index * self.chunk_time)
        self.chunk_index = (self.chunk_index + 1) % (self.sound_duration / self.chunk_time)
        return sound, pyaudio.paContinue

    def press_button_play(self) -> None:
        if not self.is_playing:
            self.is_playing = True
            self.stream.start_stream()
        else:
            self.is_playing = False
            self.stream.stop_stream()

    def shutdown_audio(self, *args):
        self.stream.close()
        self.p.terminate()
        return False
