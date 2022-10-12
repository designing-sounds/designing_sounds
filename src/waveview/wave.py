import typing

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import Graph, LinePlot
import simpleaudio as sa
import numpy as np
from src.wavemodel.wave_model import SinWave


class RootWave(BoxLayout):
    def __init__(self, **kwargs: typing.Any):
        super(RootWave, self).__init__(**kwargs)
        self.points = []
        self.num_samples = 512
        self.play.bind(on_press=self.callback)
        self.graph = Graph(border_color=[0, 1, 1, 1],
                           xmin=0, xmax=self.num_samples,
                           ymin=-1.0, ymax=1.0,
                           draw_border=False)

        self.ids.modulation.add_widget(self.graph)
        self.plot_x = np.linspace(0, 1, self.num_samples)
        self.plot_y = np.zeros(self.num_samples)
        self.plot = LinePlot(color=[1, 1, 0, 1], line_width=1.5)
        self.graph.add_plot(self.plot)
        self.update_plot(10, 100)

    def update_plot(self, freq: int, amp: int) -> None:
        sin_wave = SinWave()
        sin_wave.set_freq(freq)
        sin_wave.set_amp(amp / 100.0)
        self.points = sin_wave.get_array(self.num_samples)
        self.plot.points = [(x, self.points[x]) for x in range(self.points.size)]

    def callback(self, arg: typing.Any) -> None:
        sa.play_buffer(self.points, 2, 2, 44100)


class WaveApp(App):
    def build(self) -> RootWave:
        return RootWave()
