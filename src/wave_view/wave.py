import typing

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import LinePlot, Graph

from src.wave_model.wave_model import SoundModel
from src.wave_view.wave_sound import WaveSound
from src.wave_view.wave_graph import WaveformGraph
import numpy as np


class RootWave(BoxLayout):
    sample_rate = 44100
    graph_sample_rate = 2500
    waveform_duration = 1
    chunk_duration = 0.1
    max_samples_per_harmonic = 100
    max_harmonics = 10
    index = 1

    def __init__(self, **kwargs: typing.Any):
        super(RootWave, self).__init__(**kwargs)

        self.sound_model = SoundModel(self.max_harmonics, self.max_samples_per_harmonic)
        self.wave_sound = WaveSound(self.sample_rate, self.waveform_duration, self.chunk_duration, self.sound_model)

        self.play.bind(on_press=self.press_button_play)
        self.clear.bind(on_press=self.clear_button_play)
        self.add.bind(on_press=self.add_button_play)

        border_color = [0, 1, 1, 1]

        self.waveform_graph = WaveformGraph(update=self.update_waveform,
                                            border_color=border_color,
                                            xmin=0, xmax=self.waveform_duration,
                                            ymin=-1.0, ymax=1.0,
                                            draw_border=True, padding=0, x_grid_label=True, y_grid_label=False)
        self.power_spectrum_graph = Graph(border_color=border_color,
                                          xmin=0, xmax=500,
                                          ymin=0, ymax=1.0,
                                          draw_border=True)

        self.ids.modulation.add_widget(self.waveform_graph)
        self.ids.power_spectrum.add_widget(self.power_spectrum_graph)

        plot_color = [1, 1, 0, 1]
        self.wave_plot = LinePlot(color=plot_color, line_width=1)
        self.power_plot = LinePlot(color=plot_color, line_width=3)

        self.waveform_graph.add_plot(self.wave_plot)
        self.power_spectrum_graph.add_plot(self.power_plot)

        self.update_power_spectrum(self.mean.value, self.sd.value)

    def update_power_spectrum(self, mean: int, sd: int) -> None:
        self.power_plot.points = SoundModel.get_normal_distribution_points(mean, sd, 500)
        self.sound_model.update_power_spectrum(0, mean, sd, self.max_samples_per_harmonic)
        self.update_waveform()

    def update_waveform(self) -> None:
        inputted_points = self.waveform_graph.get_selected_points()
        self.sound_model.interpolate_points(inputted_points)
        points = self.sound_model.model_sound(self.graph_sample_rate, self.waveform_duration, 0)
        self.wave_plot.points = list(zip(np.linspace(0, self.waveform_duration, points.size), points))

    def press_button_play(self, arg: typing.Any) -> None:
        self.wave_sound.press_button_play()

    def clear_button_play(self, arg: typing.Any) -> None:
        self.waveform_graph.clear_selected_points()
        for index in range(1, self.max_harmonics):
            self.sound_model.update_power_spectrum(index, 0, 0, self.max_samples_per_harmonic)
        self.update_waveform()

    def add_button_play(self, arg: typing.Any) -> None:
        self.sound_model.update_power_spectrum(self.index, np.random.randint(100, 500), np.random.randn(),
                                               self.max_samples_per_harmonic)
        self.update_waveform()
        self.index += 1
        if self.index >= self.max_harmonics:
            self.index = 1


class WaveApp(App):
    def build(self) -> RootWave:
        return RootWave()
