import numpy as np
import pyaudio
from kivy.core.window import Window


class WaveSound:
    def __init__(self, sample_rate, time):
        self.chunk_index = 0
        self.is_playing = False
        self.sound = np.array(np.array([]))
        self.sample_rate = sample_rate
        self.time = time
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.sample_rate, output=True,
                                  stream_callback=self.callback, frames_per_buffer=int(self.sample_rate*time))
        self.stream.stop_stream()
        Window.bind(on_request_close=self.shutdown_audio)

    def callback(self, in_data, frame_count, time_info, flag):
        sound: np.ndarray = self.sound[0]
        self.chunk_index = (self.chunk_index + 1) % len(self.sound)
        return sound, pyaudio.paContinue

    def update_sound(self, sound:  np.ndarray) -> None:
        self.sound = sound

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
