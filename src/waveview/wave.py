import typing

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import Graph, LinePlot
import numpy as np

from src.wave_model.wave_model import SoundModel
from src.waveview.wave_sound import WaveSound


class RootWave(BoxLayout):
    sample_rate = 44100
    num_samples = 44100
    time = 0.1

    def __init__(self, **kwargs: typing.Any):
        super(RootWave, self).__init__(**kwargs)

        self.wave_sound = WaveSound(self.sample_rate, self.time)
        self.sound_model = SoundModel(self.sample_rate)

        self.play.bind(on_press=self.press_button_play)
        self.graph = Graph(xmin=0, xmax=self.num_samples,
                           ymin=-1.0, ymax=1.0,
                           draw_border=False)

        self.ids.modulation.add_widget(self.graph)
        self.plot = LinePlot(color=[1, 1, 0, 1], line_width=1)
        self.graph.add_plot(self.plot)
        self.update(self.freq.value, self.amp.value)

    def update(self, freq: int, amp: int):
        self.sound_model.model_sound(freq, amp, self.time)
        self.update_plot()

    def update_plot(self) -> None:
        points = self.sound_model.get_sound()
        self.plot.points = [(x / points.size * self.num_samples, points[x]) for x in range(points.size)]
        self.wave_sound.update_sound(points)

    def press_button_play(self, arg: typing.Any) -> None:
        self.wave_sound.press_button_play()


class WaveApp(App):
    def build(self) -> RootWave:
        return RootWave()
