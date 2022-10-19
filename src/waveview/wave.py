import typing

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import LinePlot

from src.wave_model.wave_model import SoundModel
from src.waveview.wave_sound import WaveSound
from src.waveview.wave_graph import RootGraph
import numpy as np


class RootWave(BoxLayout):
    sample_rate = 44100
    num_samples = 44100
    time = 1
    chunk_time = 0.1

    def __init__(self, **kwargs: typing.Any):
        super(RootWave, self).__init__(**kwargs)

        self.wave_sound = WaveSound(self.sample_rate, self.chunk_time)
        self.sound_model = SoundModel(self.sample_rate)

        self.play.bind(on_press=self.press_button_play)
        self.clear.bind(on_press=self.clear_button_play)
        self.graph = RootGraph(border_color=[0, 1, 1, 1],
                               xmin=0, xmax=self.num_samples,
                               ymin=-1.0, ymax=1.0,
                               draw_border=True)
        self.ids.modulation.add_widget(self.graph)

        self.plot = LinePlot(color=[1, 1, 0, 1], line_width=1)
        self.graph.add_plot(self.plot)

        self.update(self.freq.value, self.amp.value)

    def update(self, freq: int, amp: int):
        self.sound_model.model_sound(self.time)
        self.sound_model.normalize_sound(amp / 100)
        self.update_plot()
        self.wave_sound.update_sound(self.sound_model.reshape(self.chunk_time))

    def update_plot(self) -> None:
        points = self.sound_model.get_sound()
        self.plot.points = [(x / points.size * self.num_samples, points[x]) for x in range(points.size)]

    def press_button_play(self, arg: typing.Any) -> None:
        self.wave_sound.press_button_play()

    def clear_button_play(self, arg: typing.Any) -> None:
        self.graph.clear_selected_points()


class WaveApp(App):
    def build(self) -> RootWave:
        return RootWave()
