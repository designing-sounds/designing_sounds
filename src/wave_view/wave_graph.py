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
        x, y = self.pos
        pad_x, pad_y = self.view_pos
        pad = self.padding

        if pad_x <= touch.x - x <= self.width - pad and pad_y <= touch.y - y <= self.height - pad:
            color = (1, 1, 1)
            d = 10
            pos = (touch.x - d / 2, touch.y - d / 2)

            with self.graph_canvas.canvas:
                Color(*color, mode='hsv')
                Ellipse(pos=pos, size=(d, d))

            self.__selected_points.append(self.convert_points(pos))
            self.update()
        return super(WaveformGraph, self).on_touch_down(touch)

    def get_selected_points(self) -> typing.List[typing.Tuple[int, int]]:
        return self.__selected_points

    def clear_selected_points(self) -> None:
        self.__selected_points.clear()
        self.graph_canvas.canvas.clear()

    def convert_points(self, pos: (int, int)) -> (int, int):
        (old_x, old_y) = (max(pos[0] - self.pos[0], 0), max(pos[1] - self.pos[1], 0))

        old_range_x = self.width
        new_range_x = self.xmax - self.xmin
        new_x = (((old_x - self.xmin) * new_range_x) / old_range_x) + self.xmin

        old_range_y = self.height
        new_range_y = self.ymax - self.ymin
        new_y = (((old_y - self.ymin) * new_range_y) / old_range_y) + self.ymin
        return new_x, new_y
