import typing
from typing import Tuple, Any

import math
from kivy.graphics import Color, Ellipse, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.input.motionevent import MotionEvent
from kivy_garden.graph import Graph
from src.wave_view import style
import numpy as np

SCROLL_RIGHT = 'scrollright'

SCROLL_LEFT = 'scrollleft'

SCROLL_UP = 'scrollup'

SCROLL_DOWN = 'scrolldown'

POINT_IMAGE = 'media/black.png'


class WaveformGraph(Graph):
    __selected_points = []
    __initial_x_ticks_major = 0.1
    __point_size = 15
    __max_zoom = 100
    __min_zoom = 1
    __initial_duration = 1

    def __init__(self, update_waveform, update_waveform_graph, **kwargs):
        super().__init__(**kwargs)
        # Add kivy graph widget to canvas
        self._graph_canvas = BoxLayout(size_hint=(1, 1))
        self.add_widget(self._graph_canvas)

        # Private Class initialization
        self._update_waveform_func = update_waveform
        self._update_waveform_graph_func = update_waveform_graph
        self._last_touched_point = None
        self._zoom_scale = 1
        self._period = 500
        self.x_ticks_major = self.__initial_x_ticks_major
        self._eraser_mode = False
        self._single_period = False

        # Public Class initialization
        self.xmin = 0
        self.xmax = 1
        self.x_grid = True

    def on_touch_down(self, touch: MotionEvent) -> bool:
        a_x, a_y = self.to_widget(touch.x, touch.y, relative=True)

        if self.collide_plot(a_x, a_y):
            if touch.is_mouse_scrolling:
                if touch.button == SCROLL_DOWN:
                    self._zoom_scale = min(self._zoom_scale + 1, self.__max_zoom)
                    self.__update_zoom((a_x, a_y))
                elif touch.button == SCROLL_UP:
                    self._zoom_scale = max(self._zoom_scale - 1, self.__min_zoom)
                    self.__update_zoom((a_x, a_y))
                elif touch.button == SCROLL_LEFT:
                    self.__update_panning(False)
                elif touch.button == SCROLL_RIGHT:
                    self.__update_panning(True)
                return True

            ellipse = self.__touching_point((touch.x, touch.y))
            if ellipse:
                if self._eraser_mode:
                    self.__remove_point(ellipse)
                    touch.grab(self)
                    self._last_touched_point = None
                    return True
                self._last_touched_point = ellipse
                touch.grab(self)
                return True

            if not self._eraser_mode:
                self.__create_point(touch.pos)
                self._last_touched_point = self._graph_canvas.canvas.children[-1]
                self.__selected_points.append(tuple(self.to_data(a_x, a_y)))
                self._update_waveform_func()

        return super().on_touch_down(touch)

    def on_touch_move(self, touch: MotionEvent) -> bool:
        a_x, a_y = self.to_widget(touch.x, touch.y, relative=True)
        if self.collide_plot(a_x, a_y):
            if self._eraser_mode:
                ellipse = self.__touching_point((touch.x, touch.y))
                if ellipse:
                    self.__remove_point(ellipse)
                    self._last_touched_point = None
                    return True
                return False
            if touch.grab_current is self:
                radius = self.__point_size / 2
                ellipse = self._last_touched_point
                if ellipse is None:
                    ellipse = self.__touching_point((touch.x, touch.y))
                index = self._graph_canvas.canvas.children.index(ellipse)
                ellipse.pos = touch.x - radius, touch.y - radius
                self.__selected_points[index//3] = self.__convert_point(ellipse.pos)
                self._update_waveform_func()
                return True
        return False

    def on_touch_up(self, touch: MotionEvent) -> bool:
        if touch.grab_current is self:
            touch.ungrab(self)

        return super().on_touch_up(touch)

    def __create_point(self, touch_pos: typing.Tuple[float, float]) -> None:
        color = (0, 0, 1)
        pos = (touch_pos[0] - self.__point_size / 2, touch_pos[1] - self.__point_size / 2)
        with self._graph_canvas.canvas:
            Color(*color, mode="hsv")
            Ellipse(source=POINT_IMAGE, pos=pos, size=(self.__point_size, self.__point_size))

    def __touching_point(self, pos: typing.Tuple[float, float]) -> typing.Optional[Ellipse]:
        points = self._graph_canvas.canvas.children[2::3]
        result = None
        for point in points:
            if self.__is_inside_ellipse(point, pos):
                result = point
                break
        return result

    def __remove_point(self, ellipse):
        to_remove = self._graph_canvas.canvas.children.index(ellipse)
        self._graph_canvas.canvas.children.pop(to_remove)
        self._graph_canvas.canvas.children.pop(to_remove - 1)
        self._graph_canvas.canvas.children.pop(to_remove - 2)
        self.__selected_points.pop(to_remove//3)

        self._update_waveform_func()

    @staticmethod
    def __is_inside_ellipse(ellipse: Ellipse, pos: typing.Tuple[float, float]) -> bool:
        radius = ellipse.size[0] / 2
        x, y = (pos[0] - radius, pos[1] - radius)
        exp_x, exp_y = ellipse.pos
        return np.sqrt(np.power(exp_x - x, 2) + np.power(exp_y - y, 2)) < (ellipse.size[0] / 2)

    def __convert_point(self, point: typing.Tuple[float, float]) -> Tuple[Any, ...]:
        radius = self.__point_size / 2
        e_x, e_y = (point[0] + radius, point[1] + radius)
        a_x, a_y = self.to_widget(e_x, e_y, relative=True)
        return tuple(self.to_data(a_x, a_y))

    def get_selected_points(self) -> typing.List[typing.Tuple[float, float]]:
        return self.__selected_points

    def clear_selected_points(self) -> None:
        self.__selected_points.clear()
        self._graph_canvas.canvas.clear()
        self.__update_graph_points()

    def __to_pixels(self, data_pos: (int, int)) -> (int, int):
        (old_x, old_y) = data_pos

        old_range_x = self.xmax - self.xmin
        new_range_x = self._plot_area.size[0]
        new_x = (((old_x - self.xmin) * new_range_x) / old_range_x) + self._plot_area.pos[0] + self.x

        old_range_y = self.ymax - self.ymin
        new_range_y = self._plot_area.size[1]
        new_y = (((old_y - self.ymin) * new_range_y) / old_range_y) + self._plot_area.pos[1] + self.y
        return round(new_x), round(new_y)

    def __update_graph_points(self):
        self._graph_canvas.canvas.clear()
        for x, y in self.__selected_points:
            if self.xmin <= x <= self.xmax:
                self.__create_point(self.__to_pixels((x, y)))
        if self.xmax - self.xmin < self._period * 15:
            self.x_grid = False
            color_line = (202, 0.30, 0.85)
            current_x = self.xmin + self._period - self.xmin % self._period
            while current_x < self.xmax:
                line_x, _ = self.__to_pixels((current_x, 0))
                with self._graph_canvas.canvas:
                    Color(*color_line, mode='hsv')
                    Rectangle(pos=(line_x, self.y + self._plot_area.y), size=(2, self._plot_area.height))
                current_x += self._period
        else:
            self.x_grid = True
        self._update_waveform_graph_func()

    def __update_zoom(self, pos: typing.Tuple[float, float]) -> None:
        if not self._single_period:
            x_pos, _ = self.__convert_point(pos)
            self.x_ticks_major = self.__initial_x_ticks_major / self._zoom_scale
            left_dist = x_pos - self.xmin
            right_dist = self.xmax - x_pos
            proportion = self.__initial_duration / (left_dist + right_dist) / self._zoom_scale

            self.xmax = x_pos + proportion * right_dist
            self.xmin = x_pos - proportion * left_dist
            if self.xmin < 0:
                self.xmax -= self.xmin
                self.xmin = 0
            self.__update_graph_points()

    def __update_panning(self, is_left: bool) -> None:
        if not self._single_period:
            window_length = self.xmax - self.xmin
            factor = 1 / (self._zoom_scale * 2)
            panning_step = -factor if is_left else factor
            self.xmin += panning_step
            self.xmax += panning_step
            if self.xmin < 0:
                self.xmin = 0
                self.xmax = window_length
            self.__update_graph_points()

    def __change_period_view(self):
        if self._single_period:
            self.xmin = 0
            self.xmax = self._period
        else:
            self.xmin = 0
            self.xmax = self.__initial_duration / self._zoom_scale
        self.__update_graph_points()

    # Get/Set Methods for class
    def set_eraser_mode(self) -> None:
        self._eraser_mode = True

    def set_draw_mode(self) -> None:
        self._eraser_mode = False

    def is_eraser_mode(self) -> bool:
        return self._eraser_mode

    def set_single_period(self) -> None:
        self._single_period = True
        self.__change_period_view()

    def set_multiple_period(self) -> None:
        self._single_period = False
        self.__change_period_view()

    def is_single_period(self) -> bool:
        return self._single_period

    def set_period(self, frequency) -> None:
        if frequency != 0:
            self._period = 1 / frequency
            self.__update_graph_points()

    def get_preset_points(self, preset_func: typing.Callable, amount: int) -> typing.List[typing.Tuple[float, float]]:
        preset_wave = [(float(i), preset_func(i, self._period)) for i in np.linspace(0, self._period, amount)]
        for (i, j) in preset_wave:
            self.__create_point((i, j))
        self.__selected_points = preset_wave
        return self.__selected_points

    def get_preset_points_from_y(self, points) -> typing.List[typing.Tuple[float, float]]:
        for (i, j) in points:
            self.__create_point((i, j))
        self.__selected_points = points
        return self.__selected_points
