import typing
from typing import Tuple, Any

from kivy.graphics import Color, Ellipse
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import Graph
from kivy.input.motionevent import MotionEvent

import numpy as np


class WaveformGraph(Graph):
    __selected_points = []

    def __init__(self, update, **kwargs):
        super().__init__(**kwargs)
        self.graph_canvas = BoxLayout(size_hint=(1, 1))
        self.add_widget(self.graph_canvas)
        self.update = update
        self.current_point = None
        self.old_pos = 0
        self.d = 10

    def on_touch_down(self, touch: MotionEvent) -> bool:
        a_x, a_y = self.to_widget(touch.x, touch.y, relative=True)
        print("Coordinates to convert")
        print(touch.x, touch.y)

        if self.collide_plot(a_x, a_y):
            ellipse = self.touching_point((touch.x, touch.y))
            if ellipse:
                if touch.button == 'right':
                    to_remove = self.graph_canvas.canvas.children.index(ellipse)
                    self.graph_canvas.canvas.children.pop(to_remove)
                    self.graph_canvas.canvas.children.pop(to_remove - 1)
                    self.graph_canvas.canvas.children.pop(to_remove - 2)
                    self.__selected_points.remove(self.convert_point(ellipse.pos))
                    self.update()
                    return True
                self.current_point = ellipse
                self.old_pos = self.convert_point(self.current_point.pos)
                touch.grab(self)
                return True

            color = (0, 0, 1)

            pos = (touch.x - self.d / 2, touch.y - self.d / 2)

            with self.graph_canvas.canvas:
                Color(*color, mode='hsv')
                Ellipse(source='src/20221028_144310.jpg', pos=pos, size=(self.d, self.d))

            self.__selected_points.append(tuple(map(lambda x: round(x, 5), self.to_data(a_x, a_y))))
            self.update()

        return super(WaveformGraph, self).on_touch_down(touch)

    def on_touch_move(self, touch: MotionEvent) -> bool:
        if touch.grab_current is self:
            a_x, a_y = self.to_widget(touch.x, touch.y, relative=True)
            if self.collide_plot(a_x, a_y):
                r = self.d / 2
                self.__selected_points.remove(self.old_pos)
                self.current_point.pos = (touch.x - r, touch.y - r)
                self.old_pos = self.convert_point(self.current_point.pos)
                self.__selected_points.append(self.convert_point(self.current_point.pos))
                self.update()
            return True

    def on_touch_up(self, touch: MotionEvent) -> bool:
        if touch.grab_current is self:
            touch.ungrab(self)

        return super(WaveformGraph, self).on_touch_up(touch)

    def touching_point(self, pos: typing.Tuple[float, float]) -> typing.Optional[Ellipse]:
        points = self.graph_canvas.canvas.children[2::3]
        result = None
        for point in points:
            if self.is_inside_ellipse(point, pos):
                result = point
        return result

    @staticmethod
    def is_inside_ellipse(ellipse: Ellipse, pos: typing.Tuple[float, float]) -> bool:
        r = ellipse.size[0] / 2
        x, y = (pos[0] - r, pos[1] - r)
        exp_x, exp_y = ellipse.pos
        return np.sqrt(np.power(exp_x - x, 2) + np.power(exp_y - y, 2)) < (ellipse.size[0] / 2)

    def convert_point(self, point: typing.Tuple[float, float]) -> Tuple[Any, ...]:
        r = self.d / 2
        e_x, e_y = (point[0] + r, point[1] + r)
        a_x, a_y = self.to_widget(e_x, e_y, relative=True)
        return tuple(map(lambda x: round(x, 5), self.to_data(a_x, a_y)))

    def get_selected_points(self) -> typing.List[typing.Tuple[int, int]]:
        return self.__selected_points

    def clear_selected_points(self) -> None:
        self.__selected_points.clear()
        self.graph_canvas.canvas.clear()

    def to_pixels(self, data_pos: (int, int)) -> (int, int):
        (old_x, old_y) = data_pos

        old_range_x = self.xmax - self.xmin
        new_range_x = self._plot_area.size[0]
        new_x = (((old_x - self.xmin) * new_range_x) / old_range_x) + self._plot_area.pos[0] + self.x

        old_range_y = self.ymax - self.ymin
        new_range_y = self._plot_area.size[1]
        new_y = (((old_y - self.ymin) * new_range_y) / old_range_y) + self._plot_area.pos[1] + self.y
        return round(new_x), round(new_y)
