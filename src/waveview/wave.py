import threading
import time
import typing

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import Graph, LinePlot
import simpleaudio as sa
import numpy as np
from src.wave_model.wave_model import SinWave, normalize_sound


class RootWave(BoxLayout):
    def __init__(self, **kwargs: typing.Any):
        super(RootWave, self).__init__(**kwargs)
        self.sound = sa.PlayObject
        self.is_playing = False
        self.points = np.array([])
        self.num_samples = 44100
        self.play.bind(on_press=self.press_button_play)
        self.graph = Graph(border_color=[0, 1, 1, 1],
                           xmin=0, xmax=512,
                           ymin=-1.0, ymax=1.0,
                           draw_border=False)

        self.ids.modulation.add_widget(self.graph)
        self.plot_x = np.linspace(0, 1, self.num_samples)
        self.plot_y = np.zeros(self.num_samples)
        self.plot = LinePlot(color=[1, 1, 0, 1], line_width=1)
        starting_freq = 100
        starting_amp = 100

        self.sin_wave = SinWave(starting_freq, starting_amp / 100)
        self.graph.add_plot(self.plot)
        self.update_plot(starting_freq, starting_amp)

    def update_plot(self, freq: int, amp: int) -> None:
        self.sin_wave.freq = freq
        self.sin_wave.amp = amp / 100.0
        self.points = self.sin_wave.get_array(self.num_samples)
        self.plot.points = [(x / self.points.size * 512, self.points[x]) for x in range(self.points.size)]

    def loop_play(self) -> None:
        while self.is_playing:
            self.sound = sa.play_buffer(normalize_sound(self.points), 1, 2, 44100)
            time.sleep(len(self.points) / (44100 * 1.02))

    def press_button_play(self, arg: typing.Any) -> None:
        if not self.is_playing:
            self.is_playing = True
            my_thread = threading.Thread(target=self.loop_play)
            my_thread.start()
        else:
            self.is_playing = False
            self.sound.stop()


class WaveApp(App):
    def build(self) -> RootWave:
        return RootWave()
