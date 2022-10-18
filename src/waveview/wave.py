import typing

from kivy.app import App
from kivy.graphics import Color, Ellipse
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
        self.clear.bind(on_press=self.clear_button_play)
        self.graph = RootGraph(border_color=[0, 1, 1, 1],
                               xmin=0, xmax=self.num_samples,
                               ymin=-1.0, ymax=1.0,
                               draw_border=True)
        self.ids.modulation.add_widget(self.graph)

        self.plot = LinePlot(color=[1, 1, 0, 1], line_width=1)
        self.graph.add_plot(self.plot)
        self.sin_wave = SinWave(self.freq.value, self.amp.value / 100)
        self.update_plot(self.freq.value, self.amp.value)

    def update_plot(self, freq: int, amp: int) -> None:
        self.sin_wave.freq = freq
        self.sin_wave.amp = amp / 100.0
        self.points = self.sin_wave.get_array(self.num_samples, self.time)
        self.plot.points = [(x / self.points.size * self.num_samples, self.points[x]) for x in range(self.points.size)]
        self.wave_sound.update_sound(self.points)

    def press_button_play(self, arg: typing.Any) -> None:
        self.wave_sound.press_button_play()

    def clear_button_play(self, arg: typing.Any) -> None:
        self.graph.clear_selected_points()


class RootGraph(Graph):
    __selected_points = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph_canvas = BoxLayout(size_hint=(1, 1))
        self.add_widget(self.graph_canvas)

    def on_touch_down(self, touch) -> bool:
        graph_canvas = self.children[0]
        if self.collide_point(touch.x, touch.y):
            color = (1, 1, 1)
            d = 10
            pos = (touch.x - d / 2, touch.y - d / 2)
            with graph_canvas.canvas:
                Color(*color, mode='hsv')
                Ellipse(pos=pos, size=(d, d))

            self.__selected_points.append(self.convert_points(pos))
        return super(RootGraph, self).on_touch_down(touch)

    def get_selected_points(self) -> typing.List[(int, int)]:
        return self.__selected_points

    def clear_selected_points(self) -> None:
        self.__selected_points.clear()
        self.graph_canvas.canvas.clear()

    def convert_points(self, pos: (int, int)) -> (int, int):
        (old_x, old_y) = (max(pos[0] - self.pos[0], 0), max(pos[1] - self.pos[1], 0))
        old_range = (self.height - self.view_pos[1])
        new_range = self.ymax - self.ymin
        new_y = (((old_y - self.ymin) * new_range) / old_range) + self.ymin
        return old_x, new_y


class WaveApp(App):
    def build(self):
        return RootWave()
