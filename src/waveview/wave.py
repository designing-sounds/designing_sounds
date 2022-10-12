from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import Graph, LinePlot
import simpleaudio as sa
import numpy as np

import soundDump


class RootWave(BoxLayout):
    def __init__(self, **kwargs):
        super(RootWave, self).__init__(**kwargs)
        self.play.bind(on_press=self.callback)
        self.graph = Graph(border_color=[0, 1, 1, 1],
                           xmin=0, xmax=soundDump.soundDump.size,
                           ymin=-1.0, ymax=1.0,
                           draw_border=False)

        self.ids.modulation.add_widget(self.graph)
        self.plot_x = np.linspace(0, 1, soundDump.soundDump.size)
        self.plot_y = np.zeros(soundDump.soundDump.size)
        self.plot = LinePlot(color=[1, 1, 0, 1], line_width=1.5)
        self.graph.add_plot(self.plot)
        self.update_plot(1)

    def update_plot(self, freq):
        self.plot_y = np.sin(2 * np.pi * freq * self.plot_x)
        self.plot.points = [(x, soundDump.soundDump[x]) for x in range(soundDump.soundDump.size)]

    @staticmethod
    def callback(self):
        sa.play_buffer(soundDump.soundDump, 2, 2, 44100)


class WaveApp(App):
    def build(self):
        return RootWave()
