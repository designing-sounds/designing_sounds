import typing

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import LinePlot, Graph

from src.wave_model.wave_model import SoundModel
from src.waveview.wave_sound import WaveSound
from src.waveview.wave_graph import RootGraph
import numpy as np


class RootWave(BoxLayout):
    sample_rate = 44100
    num_samples = 44100
    time = 1

    def __init__(self, **kwargs: typing.Any):
        super(RootWave, self).__init__(**kwargs)

        self.wave_sound = WaveSound(self.sample_rate, self.time)
        self.sound_model = SoundModel(self.sample_rate)

        self.play.bind(on_press=self.press_button_play)
        self.clear.bind(on_press=self.clear_button_play)
        self.waveform_graph = RootGraph(border_color=[0, 1, 1, 1],
                                        xmin=0, xmax=self.num_samples,
                                        ymin=-1.0, ymax=1.0,
                                        draw_border=True)
        self.power_spectrum_graph = Graph(border_color=[0, 1, 1, 1],
                                          xmin=0, xmax=self.num_samples,
                                          ymin=-1.0, ymax=1.0,
                                          draw_border=True)

        self.ids.modulation.add_widget(self.waveform_graph)
        self.ids.power_spectrum.add_widget(self.power_spectrum_graph)

        self.wave_plot = LinePlot(color=[1, 1, 0, 1], line_width=1)
        self.power_plot = LinePlot(color=[1, 1, 0, 1], line_width=1)

        self.waveform_graph.add_plot(self.wave_plot)
        self.power_spectrum_graph.add_plot(self.power_plot)
        self.update(100, self.amplitude.value)

    def update(self, freq: int, amp: int) -> None:
        self.sound_model.model_sound(self.time)
        self.sound_model.normalize_sound(amp / 100)
        self.update_plot()
        self.wave_sound.update_sound(np.reshape(self.sound_model.get_sound(), (1, -1)))

    def update_power_spectrum(self, sd: int, offset: int, amplitude: int) -> None:
        pass
        # update plot of power spectrum with new normal distribution
        # update main graph and sound possibly calling other update function

    def update_plot(self) -> None:
        points = self.sound_model.get_sound()
        self.wave_plot.points = [(x / points.size * self.num_samples, points[x]) for x in range(points.size)]
        self.power_plot.points = [(x / points.size * self.num_samples, points[x]) for x in range(points.size)]

    def press_button_play(self, arg: typing.Any) -> None:
        self.wave_sound.press_button_play()

    def clear_button_play(self, arg: typing.Any) -> None:
        self.waveform_graph.clear_selected_points()


class WaveApp(App):
    def build(self) -> RootWave:
        return RootWave()
