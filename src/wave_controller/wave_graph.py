import typing
from typing import Tuple, Optional, List

from kivy.graphics import Color, Ellipse, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.input.motionevent import MotionEvent
from kivy_garden.graph import Graph
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
        self._zoom_scale = self.__min_zoom
        self._period = 0.002
        self.x_ticks_major = self.__initial_x_ticks_major
        self._eraser_mode = False
        self._is_single_period = False

        # Public Class initialization
        self.xmin = 0
        self.xmax = 1
        self.x_grid = True

    def on_touch_down(self, touch: MotionEvent) -> bool:
        a_x, a_y = self.to_widget(touch.x, touch.y, relative=True)

        if self.collide_plot(a_x, a_y):
            if touch.is_mouse_scrolling:
                if touch.button == SCROLL_DOWN:
                    self._zoom_scale = min(self._zoom_scale + (1 / self._period) / 100, (1 / self._period) / 2)
                    self.__update_zoom((a_x, a_y), True)
                elif touch.button == SCROLL_UP:
                    self._zoom_scale = max(self._zoom_scale - (1 / self._period) / 100, self.__min_zoom)
                    self.__update_zoom((a_x, a_y), False)
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
                self._last_touched_point = self.__create_point(touch.pos)
                self.__selected_points.append([tuple(self.to_data(a_x, a_y)), self._last_touched_point])
                self._update_waveform_func(update_noise=True)

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

                ellipse.pos = touch.x - radius, touch.y - radius
                point, _ = self.get_point_from_ellipse(ellipse)
                point[0] = self.__convert_point(ellipse.pos)
                self._update_waveform_func(update_noise=True)
                return True
        return False

    def on_touch_up(self, touch: MotionEvent) -> bool:
        if touch.grab_current is self:
            touch.ungrab(self)

        return super().on_touch_up(touch)

    def get_point_from_ellipse(self, ellipse: Ellipse) -> Tuple[Ellipse, int]:
        for i, point in enumerate(self.__selected_points):
            if ellipse == point[1]:
                return point, i
        return None, None

    def __create_point(self, touch_pos: Tuple[float, float]) -> Ellipse:
        color = (0, 0, 1)
        pos = (touch_pos[0] - self.__point_size / 2, touch_pos[1] - self.__point_size / 2)
        with self._graph_canvas.canvas:
            Color(*color, mode="hsv")
            Ellipse(source=POINT_IMAGE, pos=pos, size=(self.__point_size, self.__point_size))

        return self._graph_canvas.canvas.children[-1]

    def __touching_point(self, pos: typing.Tuple[float, float]) -> Optional[Ellipse]:
        points = self._graph_canvas.canvas.children[2::3]
        for point in points:
            if self.__is_inside_ellipse(point, pos):
                return point
        return None

    def __remove_point(self, ellipse: Ellipse):
        to_remove = self._graph_canvas.canvas.children.index(ellipse)
        self._graph_canvas.canvas.children.pop(to_remove)
        self._graph_canvas.canvas.children.pop(to_remove - 1)
        self._graph_canvas.canvas.children.pop(to_remove - 2)
        _, index = self.get_point_from_ellipse(ellipse)
        self.__selected_points.pop(index)

        self._update_waveform_func(update_noise=True)

    @staticmethod
    def __is_inside_ellipse(ellipse: Ellipse, pos: Tuple[float, float]) -> bool:
        radius = ellipse.size[0] / 2
        x, y = (pos[0] - radius, pos[1] - radius)
        exp_x, exp_y = ellipse.pos
        return np.sqrt(np.power(exp_x - x, 2) + np.power(exp_y - y, 2)) < (ellipse.size[0] / 2)

    def __convert_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        radius = self.__point_size / 2
        e_x, e_y = (point[0] + radius, point[1] + radius)
        a_x, a_y = self.to_widget(e_x, e_y, relative=True)
        return tuple(self.to_data(a_x, a_y))

    def get_selected_points(self) -> List[Tuple[float, float]]:
        return [x for x, _ in self.__selected_points]

    def clear_selected_points(self) -> None:
        self._graph_canvas.canvas.clear()
        self.__selected_points.clear()
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
        self._redraw_all()
        for point in self.__selected_points:
            x, y = point[0]
            if self.xmin <= x <= self.xmax:
                point[1] = self.__create_point(self.__to_pixels((x, y)))
        if self.xmax - self.xmin < self._period * 15:
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

    def __update_zoom(self, pos: Tuple[float, float], zoom_in: bool) -> None:
        x_pos, _ = self.__convert_point(pos)
        if zoom_in and self.xmax - self.xmin < self._period * 3:
            self._update_single_period(x_pos)
            self._is_single_period = True
        else:
            self.x_ticks_major = self.__initial_x_ticks_major / self._zoom_scale
            left_dist = x_pos - self.xmin
            right_dist = self.xmax - x_pos
            proportion = self.__initial_duration / (left_dist + right_dist) / self._zoom_scale

            self.xmax = x_pos + proportion * right_dist
            self.xmin = x_pos - proportion * left_dist
            self._is_single_period = False
        if self.xmin < 0:
            self.xmax -= self.xmin
            self.xmin = 0
        self.__update_graph_points()

    def __update_panning(self, is_left: bool) -> None:
        window_length = self.xmax - self.xmin
        factor = 1 / (self._zoom_scale * 5)
        panning_step = -factor if is_left else factor
        if window_length < self._period * 2:
            panning_step = -self._period if is_left else self._period
        self.xmin += panning_step
        self.xmax += panning_step
        if self.xmin < 0:
            self.xmin = 0
            self.xmax = window_length
        self.__update_graph_points()

    def _update_single_period(self, x_pos: float):
        self.xmin = (x_pos // self._period) * self._period
        self.xmax = self.xmin + self._period
        self.x_ticks_major = self._period / 4

    # Get/Set Methods for class
    def set_eraser_mode(self) -> None:
        self._eraser_mode = True

    def set_draw_mode(self) -> None:
        self._eraser_mode = False

    def is_eraser_mode(self) -> bool:
        return self._eraser_mode

    def set_period(self, frequency: float) -> None:
        if frequency == 0:
            return
        new_period = 1 / frequency
        if new_period != self._period:
            self._period = new_period
            pos = ((self.xmax - self.xmin) / 2 + self.xmin, 0)
            if self._is_single_period:
                x_pos, _ = self.__convert_point(pos)
                self._update_single_period(x_pos)
                self._zoom_scale = (1 / self._period) / 2
                if self.xmin < 0:
                    self.xmax -= self.xmin
                    self.xmin = 0
                self.__update_graph_points()
            else:
                self.__update_zoom(pos, False)

    def fit_to_new_frequency(self, old_frequency, new_frequency):
        if old_frequency != 0:
            scale = old_frequency / new_frequency
            for point in self.__selected_points:
                point[0] = (point[0][0] * scale, point[0][1])
                point[1].pos = (point[1].pos[0] * scale, point[1].pos[1])
            self.__update_graph_points()

    def get_preset_points(self, preset_func: typing.Callable, amount: int, square: bool, sawtooth: bool) -> List[Tuple[float, float]]:
        points = []
        spaced = np.linspace(0, self._period, amount)
        for i in spaced:
            points.append((float(i), preset_func(i, self._period)))
        if square or sawtooth:
            points.pop(0)
            points.pop(-1)
        if square:
            points.pop(amount // 2)
            points.pop(amount // 2 - 1)
        return self.get_preset_points_from_y(points)

    def get_preset_points_from_y(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        self.clear_selected_points()

        for point in points:
            self.__selected_points.append([point, self.__create_point(self.__to_pixels(point))])
        self._update_waveform_func(update_noise=True)
        self.__update_graph_points()
        return self.get_selected_points()
