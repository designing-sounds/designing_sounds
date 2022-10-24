from threading import Thread
import typing

from multiprocessing import Process

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import LinePlot, Graph

from src.wave_model.wave_model import SoundModel, PowerSpectrum
from src.waveview.wave_sound import WaveSound
from src.waveview.wave_graph import RootGraph
import numpy as np


class RootWave(BoxLayout):
    sample_rate = 44100
    num_samples = 44100
    time = 5
    chunk_time = 0.1

    def __init__(self, **kwargs: typing.Any):
        super(RootWave, self).__init__(**kwargs)

        self.wave_sound = WaveSound(self.sample_rate, self.chunk_time)
        self.sound_model = SoundModel(self.sample_rate)

        self.play.bind(on_press=self.press_button_play)
        self.clear.bind(on_press=self.clear_button_play)
        self.waveform_graph = RootGraph(border_color=[0, 1, 1, 1],
                                        xmin=0, xmax=self.time,
                                        ymin=-1.0, ymax=1.0,
                                        draw_border=True)
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

        self.t = None

        self.newsd = 0
        self.newoffset = 0

        self.update_with_thread(self.sd.value, self.offset.value)

    def update(self) -> None:
        self.sound_model.model_sound(self.time)
        self.sound_model.normalize_sound()
        self.update_plot()
        self.wave_sound.update_sound(self.sound_model.reshape(self.chunk_time))

    def update_with_thread(self, sd: int, offset: int, *args, **kwargs) -> None:
        self.newsd = sd
        self.newoffset = offset
        if not self.t:
            self.t = Thread(target = self.run_update_thread)
            self.t.start()
        else:
            self.t._target = self.run_update_thread
            self.t._args = args
            self.t._kwargs = kwargs
            self.t.run()

    def run_update_thread(self):
        self.update_power_spectrum(self.newsd, self.newoffset)

    def update_power_spectrum(self, sd: int, offset: int) -> None:
        xs, ys = PowerSpectrum.get_normal_distribution_points(offset, sd, 500)
        self.power_plot.points = list(zip(xs, ys))
        self.sound_model.update_power_spectrum(np.reshape(np.array([offset, sd]), (1, -1)))

        self.update()
        # update plot of power spectrum with new normal distribution
        # update main graph and sound possibly calling other update function

    def update_plot(self) -> None:
        points = self.sound_model.get_sound()
        self.wave_plot.points = list(zip(np.linspace(0, self.time, points.size // self.time), points))

    def press_button_play(self, arg: typing.Any) -> None:
        self.wave_sound.press_button_play()

    def clear_button_play(self, arg: typing.Any) -> None:
        self.waveform_graph.clear_selected_points()


class WaveApp(App):
    def build(self) -> RootWave:
        return RootWave()
