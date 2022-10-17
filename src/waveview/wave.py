import typing

from kivy.app import App
from kivy.graphics import Color, Ellipse
from kivy.config import Config
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import Graph, LinePlot
import numpy as np

from src.wave_model.wave_model import SinWave
from src.waveview.wave_sound import WaveSound


class RootWave(BoxLayout):
    sample_rate = 44100
    num_samples = 44100
    time = 0.1

    def __init__(self, **kwargs: typing.Any):
        super(RootWave, self).__init__(**kwargs)
        self.points = np.array([])
        self.wave_sound = WaveSound(self.sample_rate, self.time)

        self.play.bind(on_press=self.press_button_play)
        self.graph = RootGraph(border_color=[0, 1, 1, 1],
                               xmin=0, xmax=self.num_samples,
                               ymin=-1.0, ymax=1.0,
                               draw_border=True)

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
        self.points = self.sin_wave.get_array(self.num_samples, self.time)
        self.plot.points = [(x / self.points.size * self.num_samples, self.points[x]) for x in range(self.points.size)]
        self.wave_sound.update_sound(self.points)

    def press_button_play(self, arg: typing.Any) -> None:
        self.wave_sound.press_button_play()


class RootGraph(Graph):

    def on_touch_down(self, touch):
        if self.collide_point(touch.x, touch.y):
            x, y = self.to_widget(touch.x, touch.y)
            color = (1, 1, 1)
            with self.canvas:
                Color(*color, mode='hsv')
                d = 10
                Ellipse(pos=(x - d / 2, y - d / 2), size=(d, d))

        return super(RootGraph, self).on_touch_down(touch)


class WaveApp(App):
    def build(self) -> RootWave:
        return RootWave()
