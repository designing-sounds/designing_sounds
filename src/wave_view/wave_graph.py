import typing

from kivy.graphics import Color, Ellipse, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scatterlayout import ScatterLayout
from kivy_garden.graph import Graph

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
        self.d = 15

    def on_touch_down(self, touch) -> bool:
        a_x, a_y = self.to_widget(touch.x, touch.y, relative=True)
        if self.collide_plot(a_x, a_y):

            ellipse = self.touching_point((touch.x, touch.y))

            if ellipse:
                self.current_point = ellipse
                r = self.d / 2
                e_x, e_y = (self.current_point.pos[0] + r, self.current_point.pos[1] + r)
                a_x, a_y = self.to_widget(e_x, e_y, relative=True)
                self.old_pos = list(map(lambda x: round(x, 6), self.to_data(a_x, a_y)))
                touch.grab(self)
                return True
            color = (1, 1, 1)

            pos = (touch.x - self.d / 2, touch.y - self.d / 2)

            with self.graph_canvas.canvas:
                Color(*color, mode='hsv')
                Ellipse(source='src/20221028_144310.jpg', pos=pos, size=(self.d, self.d))

            self.__selected_points.append(list(map(lambda x: round(x, 6), self.to_data(a_x, a_y))))

            self.update()

        return super(WaveformGraph, self).on_touch_down(touch)

    def on_touch_move(self, touch) -> bool:
        if touch.grab_current is self:
            a_x, a_y = self.to_widget(touch.x, touch.y, relative=True)
            if self.collide_plot(a_x, a_y):
                r = self.d / 2
                self.current_point.pos = (touch.x - r, touch.y - r)
            return True

    def on_touch_up(self, touch) -> bool:
        if touch.grab_current is self:
            touch.ungrab(self)
            print(self.old_pos)
            self.__selected_points.remove(self.old_pos)
            r = self.d / 2
            e_x, e_y = (self.current_point.pos[0] + r, self.current_point.pos[1] + r)
            a_x, a_y = self.to_widget(e_x, e_y, relative=True)
            self.__selected_points.append(list(map(lambda x: round(x, 6), self.to_data(a_x, a_y))))
            self.update()
        return super(WaveformGraph, self).on_touch_up(touch)

    def touching_point(self, pos: typing.Tuple[int, int]) -> typing.Optional[Ellipse]:
        points = self.graph_canvas.canvas.children[2::3]
        print(points)
        result = None
        for point in points:
            if self.is_touching_point(point, pos):
                result = point
        return result

    def is_touching_point(self, ellipse: Ellipse, pos: typing.Tuple[int, int]) -> bool:
        r = ellipse.size[0] / 2
        x, y = (pos[0] - r, pos[1] - r)
        exp_x, exp_y = ellipse.pos
        return np.sqrt(np.power(exp_x - x, 2) + np.power(exp_y - y, 2)) < (ellipse.size[0] / 2)

    def get_selected_points(self) -> typing.List[typing.Tuple[int, int]]:
        return self.__selected_points

    def clear_selected_points(self) -> None:
        self.__selected_points.clear()
        self.graph_canvas.canvas.clear()
