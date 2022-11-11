import typing
from typing import Tuple, Any

import math
from kivy.graphics import Color, Ellipse
from kivy.uix.boxlayout import BoxLayout
from kivy.input.motionevent import MotionEvent
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
        self.old_pos = None
        self.point_size = 15
        self.old_x = None
        self.panning_mode = False
        self.xmin = 0
        self.xmax = 1

    def on_touch_down(self, touch: MotionEvent) -> bool:
        a_x, a_y = self.to_widget(touch.x, touch.y, relative=True)

        if self.collide_plot(a_x, a_y):
            if self.panning_mode:
                self.old_x, _ = self.convert_point((a_x, a_y))
                touch.grab(self)
                return True
            ellipse = self.touching_point((touch.x, touch.y))
            if ellipse:
                if touch.button == 'right':
                    to_remove = self.graph_canvas.canvas.children.index(ellipse)
                    self.graph_canvas.canvas.children.pop(to_remove)
                    self.graph_canvas.canvas.children.pop(to_remove - 1)
                    self.graph_canvas.canvas.children.pop(to_remove - 2)
                    x, y = self.convert_point(ellipse.pos)
                    for point in self.__selected_points:
                        if math.isclose(point[0], x, abs_tol=0.001) and point[1] == y:
                            self.__selected_points.remove(point)
                            break
                    self.update()
                    return True
                self.current_point = ellipse
                self.old_pos = self.convert_point(self.current_point.pos)
                touch.grab(self)
                return True

            color = (0, 0, 1)

            pos = (touch.x - self.point_size / 2, touch.y - self.point_size / 2)

            with self.graph_canvas.canvas:
                Color(*color, mode='hsv')
                Ellipse(source='media/20221028_144310.jpg', pos=pos, size=(self.point_size, self.point_size))

            self.__selected_points.append(tuple(map(lambda x: round(x, 5), self.to_data(a_x, a_y))))
            self.update()

        return super().on_touch_down(touch)

    def on_touch_move(self, touch: MotionEvent) -> bool:
        from src.wave_controller.wave import RootWave
        if touch.grab_current is self:
            a_x, a_y = self.to_widget(touch.x, touch.y, relative=True)
            if self.collide_plot(a_x, a_y):
                if self.panning_mode:
                    total_duration = RootWave.waveform_duration
                    new_x, _ = self.convert_point((a_x, a_y))
                    window_length = self.xmax - self.xmin
                    self.xmin += self.old_x - new_x
                    self.xmax += self.old_x - new_x
                    if self.xmax > total_duration:
                        self.xmax = total_duration
                        self.xmin = self.xmax - window_length
                    elif self.xmin < 0:
                        self.xmin = 0
                        self.xmax = window_length
                    self.xmin = round(self.xmin, 3)
                    self.old_x = new_x
                    self.update_graph_points()
                    return True
                radius = self.point_size / 2
                for point in self.__selected_points:
                    if math.isclose(point[0], self.old_pos[0], abs_tol=0.001) and point[1] == self.old_pos[1]:
                        self.__selected_points.remove(point)
                        break
                self.current_point.pos = (touch.x - radius, touch.y - radius)
                self.old_pos = self.convert_point(self.current_point.pos)
                self.__selected_points.append(self.convert_point(self.current_point.pos))
                self.update()
            return True
        return False

    def on_touch_up(self, touch: MotionEvent) -> bool:
        if touch.grab_current is self:
            touch.ungrab(self)

        return super().on_touch_up(touch)

    def touching_point(self, pos: typing.Tuple[float, float]) -> typing.Optional[Ellipse]:
        points = self.graph_canvas.canvas.children[2::3]
        result = None
        for point in points:
            if self.is_inside_ellipse(point, pos):
                result = point
        return result

    @staticmethod
    def is_inside_ellipse(ellipse: Ellipse, pos: typing.Tuple[float, float]) -> bool:
        radius = ellipse.size[0] / 2
        x, y = (pos[0] - radius, pos[1] - radius)
        exp_x, exp_y = ellipse.pos
        return np.sqrt(np.power(exp_x - x, 2) + np.power(exp_y - y, 2)) < (ellipse.size[0] / 2)

    def convert_point(self, point: typing.Tuple[float, float]) -> Tuple[Any, ...]:
        radius = self.point_size / 2
        e_x, e_y = (point[0] + radius, point[1] + radius)
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

    def update_graph_points(self):
        self.graph_canvas.canvas.clear()
        for x, y in self.__selected_points:
            if self.xmin <= x <= self.xmax:
                new_x, new_y = self.to_pixels((x, y))
                color = (0, 0, 1)
                pos = (new_x - self.point_size / 2, new_y - self.point_size / 2)
                with self.graph_canvas.canvas:
                    Color(*color, mode='hsv')
                    Ellipse(source='media/20221028_144310.jpg', pos=pos,
                            size=(self.point_size, self.point_size))
        self.update()
