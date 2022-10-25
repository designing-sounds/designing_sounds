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
    time = 2
    chunk_time = 0.1

    def __init__(self, **kwargs: typing.Any):
        super(RootWave, self).__init__(**kwargs)

        self.sound_model = SoundModel()
        self.wave_sound = WaveSound(self.sample_rate, self.time, self.chunk_time, self.sound_model)

        self.play.bind(on_press=self.press_button_play)
        self.clear.bind(on_press=self.clear_button_play)
        self.waveform_graph = RootGraph(border_color=[0, 1, 1, 1],
                                        xmin=0, xmax=self.time,
                                        ymin=-1.0, ymax=1.0,
                                        draw_border=True, padding=0, x_grid_label=True, y_grid_label=False)
        self.power_spectrum_graph = Graph(border_color=[0, 1, 1, 1],
                                          xmin=0, xmax=500,
                                          ymin=0, ymax=1.0,
                                          draw_border=True)

        self.ids.modulation.add_widget(self.waveform_graph)
        self.ids.power_spectrum.add_widget(self.power_spectrum_graph)

        self.wave_plot = LinePlot(color=[1, 1, 0, 1], line_width=1)
        self.power_plot = LinePlot(color=[1, 1, 0, 1], line_width=3)

        self.waveform_graph.add_plot(self.wave_plot)
        self.power_spectrum_graph.add_plot(self.power_plot)

        harmonic = self.sound_model.add_to_power_spectrum()
        self.harmonics = [harmonic]
        self.update_power_spectrum(self.sd.value, self.offset.value)

    def update_power_spectrum(self, sd: int, offset: int) -> None:
        self.power_plot.points = SoundModel.get_normal_distribution_points(offset, sd, 500)
        self.sound_model.update_power_spectrum(self.harmonics[0], offset, sd, 250)

        self.update_plot()
        # update plot of power spectrum with new normal distribution
        # update main graph and sound possibly calling other update function

    def update_plot(self) -> None:
        sample_rate = 2000
        points = self.sound_model.model_sound(sample_rate, self.time, 0)
        self.wave_plot.points = list(zip(np.linspace(0, self.time, points.size), points))

    def press_button_play(self, arg: typing.Any) -> None:
        self.wave_sound.press_button_play()

    def clear_button_play(self, arg: typing.Any) -> None:
        self.waveform_graph.clear_selected_points()


class WaveApp(App):
    def build(self) -> RootWave:
        return RootWave()
