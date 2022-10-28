import typing

from kivy.graphics import Color, Ellipse
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import Graph


class WaveformGraph(Graph):
    __selected_points = []

    def __init__(self, update, **kwargs):
        super().__init__(**kwargs)
        self.graph_canvas = BoxLayout(size_hint=(1, 1))
        self.add_widget(self.graph_canvas)
        self.update = update

    def on_touch_down(self, touch) -> bool:
        a_x, a_y = self.to_widget(touch.x, touch.y, relative=True)
        if self.collide_plot(a_x, a_y):
            color = (1, 1, 1)
            d = 10
            pos = (touch.x - d / 2, touch.y - d / 2)

            with self.graph_canvas.canvas:
                Color(*color, mode='hsv')
                Ellipse(pos=pos, size=(d, d))

            self.__selected_points.append(self.to_data(a_x, a_y))
            self.update()
        return super(WaveformGraph, self).on_touch_down(touch)

    def get_selected_points(self) -> typing.List[typing.Tuple[int, int]]:
        return self.__selected_points

    def clear_selected_points(self) -> None:
        self.__selected_points.clear()
        self.graph_canvas.canvas.clear()

