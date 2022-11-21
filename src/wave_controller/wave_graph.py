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

    def __init__(self, update_waveform, update_waveform_graph, update_power_spectrum_graph, **kwargs):
        super().__init__(**kwargs)
        self.graph_canvas = BoxLayout(size_hint=(1, 1))
        self.add_widget(self.graph_canvas)
        self.update_waveform = update_waveform
        self.update_waveform_graph = update_waveform_graph
        self.update_power_spectrum_graph = update_power_spectrum_graph
        self.current_point = None
        self.old_pos = None
        self.point_size = 15
        self.old_x = None
        self.initial_duration = 1
        self.xmin = 0
        self.xmax = 1
        self.max_zoom = 100
        self.min_zoom = 1
        self.zoom_scale = 1
        self.initial_x_ticks_major = 0.1
        self.x_ticks_major = self.initial_x_ticks_major
        self.eraser_mode = False

    def on_touch_down(self, touch: MotionEvent) -> bool:
        a_x, a_y = self.to_widget(touch.x, touch.y, relative=True)

        if self.collide_plot(a_x, a_y):
            if touch.is_mouse_scrolling:
                if touch.button == 'scrolldown':
                    self.zoom_scale = min(self.zoom_scale + 1, self.max_zoom)
                    self.update_zoom((a_x, a_y))
                elif touch.button == 'scrollup':
                    self.zoom_scale = max(self.zoom_scale - 1, self.min_zoom)
                    self.update_zoom((a_x, a_y))
                elif touch.button == 'scrollleft':
                    self.update_panning(False)
                elif touch.button == 'scrollright':
                    self.update_panning(True)
                return True

            ellipse = self.touching_point((touch.x, touch.y))
            if ellipse:
                if self.eraser_mode:
                    self.remove_point(ellipse)
                    touch.grab(self)
                    return True
                self.current_point = ellipse
                self.old_pos = self.convert_point(self.current_point.pos)
                touch.grab(self)
                return True

            if not self.eraser_mode:
                color = (0, 0, 1)

                pos = (touch.x - self.point_size / 2, touch.y - self.point_size / 2)

                with self.graph_canvas.canvas:
                    Color(*color, mode='hsv')
                    Ellipse(source='media/20221028_144310.jpg', pos=pos, size=(self.point_size, self.point_size))

                self.__selected_points.append(tuple(map(lambda x: round(x, 5), self.to_data(a_x, a_y))))
                self.update_waveform()
                self.update_power_spectrum_graph()

        return super().on_touch_down(touch)

    def on_touch_move(self, touch: MotionEvent) -> bool:
        a_x, a_y = self.to_widget(touch.x, touch.y, relative=True)
        if self.collide_plot(a_x, a_y):

            if self.eraser_mode:
                ellipse = self.touching_point((touch.x, touch.y))
                if ellipse:
                    self.remove_point(ellipse)
                    return True

            if touch.grab_current is self:
                radius = self.point_size / 2
                for point in self.__selected_points:
                    if math.isclose(point[0], self.old_pos[0], abs_tol=0.001) and point[1] == self.old_pos[1]:
                        self.__selected_points.remove(point)
                        break
                self.current_point.pos = (touch.x - radius, touch.y - radius)
                self.old_pos = self.convert_point(self.current_point.pos)
                self.__selected_points.append(self.convert_point(self.current_point.pos))
                self.update_waveform()
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
                break
        return result

    def remove_point(self, ellipse):
        to_remove = self.graph_canvas.canvas.children.index(ellipse)
        self.graph_canvas.canvas.children.pop(to_remove)
        self.graph_canvas.canvas.children.pop(to_remove - 1)
        self.graph_canvas.canvas.children.pop(to_remove - 2)
        x, y = self.convert_point(ellipse.pos)
        for point in self.__selected_points:
            if math.isclose(point[0], x, abs_tol=0.001) and point[1] == y:
                self.__selected_points.remove(point)
                break
        self.update_waveform()

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
        self.update_waveform_graph()

    def update_zoom(self, pos: typing.Tuple[float, float]) -> None:
        x_pos, _ = self.convert_point(pos)
        self.x_ticks_major = self.initial_x_ticks_major / self.zoom_scale
        left_dist = x_pos - self.xmin
        right_dist = self.xmax - x_pos
        proportion = self.initial_duration / (left_dist + right_dist) / self.zoom_scale

        self.xmax = x_pos + proportion * right_dist
        self.xmin = x_pos - proportion * left_dist
        if self.xmin < 0:
            self.xmax -= self.xmin
            self.xmin = 0
        self.update_graph_points()

    def update_panning(self, is_left: bool) -> None:
        window_length = self.xmax - self.xmin
        factor = 1 / (self.zoom_scale * 2)
        panning_step = -factor if is_left else factor
        self.xmin += panning_step
        self.xmax += panning_step
        if self.xmin < 0:
            self.xmin = 0
            self.xmax = window_length
        self.update_graph_points()
