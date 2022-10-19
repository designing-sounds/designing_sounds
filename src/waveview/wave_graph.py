import typing

from kivy.graphics import Color, Ellipse
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import Graph


class RootGraph(Graph):
    __selected_points = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph_canvas = BoxLayout(size_hint=(1, 1))
        self.add_widget(self.graph_canvas)

    def on_touch_down(self, touch) -> bool:
        if self.collide_point(touch.x, touch.y):
            color = (1, 1, 1)
            d = 10
            pos = (touch.x - d / 2, touch.y - d / 2)
            with self.graph_canvas.canvas:
                Color(*color, mode='hsv')
                Ellipse(pos=pos, size=(d, d))

            self.__selected_points.append(self.convert_points(pos))
        return super(RootGraph, self).on_touch_down(touch)

    def get_selected_points(self) -> typing.List[typing.Tuple[int, int]]:
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