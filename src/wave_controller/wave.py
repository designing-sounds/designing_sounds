import typing

import numpy as np
from kivy.lang import Builder
from kivy_garden.graph import LinePlot, Graph, BarPlot
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRectangleFlatButton

from src.wave_controller.wave_graph import WaveformGraph
from src.wave_controller.wave_sound import WaveSound
from src.wave_model.wave_model import SoundModel
from src.wave_view import style

Builder.load_file('src/wave_view/wave.kv')


class RootWave(MDBoxLayout):
    sample_rate = 44100
    graph_sample_rate = 2500
    power_spectrum_graph_samples = 1000
    waveform_duration = 1
    chunk_duration = 0.1

    max_harmonics = 10
    num_power_spectrums = 0
    current_harmonic_index = 0
    max_samples_per_harmonic = 100

    def __init__(self, **kwargs: typing.Any):
        super().__init__(**kwargs)

        self.sound_model = SoundModel(self.max_harmonics, self.max_samples_per_harmonic, int(self.mean.max))
        self.wave_sound = WaveSound(self.sample_rate, self.waveform_duration, self.chunk_duration, self.sound_model)

        # Button bindings
        self.play.bind(on_press=self.press_button_play)
        self.back.bind(on_press=self.press_button_back)
        self.eraser_mode.bind(on_press=self.press_button_eraser)
        self.clear.bind(on_press=self.press_button_clear)
        self.add.bind(on_press=self.press_button_add)
        self.all_power_spectrums.bind(on_press=self.press_button_all_power_spectrum)
        self.power_spectrum_sliders = [self.sd, self.mean, self.harmonic_samples, self.num_harmonics,
                                       self.decay_function]

        border_color = [0, 0, 0, 1]
        self.waveform_graph = WaveformGraph(update_waveform=self.update_waveform,
                                            update_waveform_graph=self.update_waveform_graph, size_hint=(1, 1),
                                            border_color=border_color,
                                            xmin=0, xmax=self.waveform_duration, ymin=-1.0, ymax=1.0, padding=10,
                                            draw_border=True, x_grid_label=True, y_grid_label=True, xlabel='Time',
                                            ylabel='Amplitude', precision="%.5g", x_grid=True, y_grid=True,
                                            y_ticks_major=0.25, label_options=dict(color=(0, 0, 0, 1)))
        self.power_spectrum_graph = Graph(border_color=border_color,
                                          xmin=0, xmax=self.mean.max, ymin=0, ymax=20, padding=10,
                                          x_grid_label=True, y_grid_label=True, xlabel='Frequency (Hz)',
                                          x_ticks_major=100, y_ticks_major=10, y_ticks_minor=5, tick_color=(1, 0, 0, 0),
                                          label_options=dict(color=(0, 0, 0, 1)))

        self.ids.modulation.add_widget(self.waveform_graph)
        self.ids.power_spectrum.add_widget(self.power_spectrum_graph)

        plot_color = style.cyber_grape

        self.wave_plot = LinePlot(color=plot_color, line_width=1)
        self.power_plot = BarPlot(color=plot_color)

        self.waveform_graph.add_plot(self.wave_plot)
        self.power_spectrum_graph.add_plot(self.power_plot)

        self.power_buttons = []
        self.selected_button_color = style.dark_sky_blue
        self.unselected_button_color = style.blue_violet
        self.harmonic_list = [[0, 0, 0, 1, "1 / x"]] * self.max_harmonics
        self.press_button_add(None)
        self.double_tap = False
        self.change_power_spectrum = True

    def update_power_spectrum(self) -> None:
        if self.change_power_spectrum:
            harmonic_samples = int(self.max_samples_per_harmonic * self.harmonic_samples.value / 100)
            self.sound_model.update_power_spectrum(self.current_harmonic_index, self.mean.value, self.sd.value,
                                                   harmonic_samples, int(self.num_harmonics.value),
                                                   self.decay_function.text)
            self.update_power_spectrum_graph()
            self.update_waveform()

    def update_power_spectrum_graph(self):
        self.power_plot.points = self.sound_model.get_power_spectrum_histogram(self.current_harmonic_index,
                                                                               self.power_spectrum_graph_samples)
        self.power_spectrum_graph.ymax = max(int(max(self.power_plot.points, key=lambda x: x[1])[1]), 1)

    def update_waveform(self) -> None:
        self.sound_model.interpolate_points(self.waveform_graph.get_selected_points())
        self.wave_sound.sound_changed()
        self.update_waveform_graph()

    def update_waveform_graph(self) -> None:
        x_min = self.waveform_graph.xmin
        x_max = self.waveform_graph.xmax
        points = self.sound_model.model_sound(self.graph_sample_rate / (x_max - x_min), x_max - x_min, x_min)
        self.wave_plot.points = list(zip(np.linspace(x_min, x_max, points.size), points))

    def press_button_play(self, _: typing.Any) -> None:
        if self.wave_sound.is_playing():
            self.wave_sound.pause_audio()
            self.play.icon = "play"
            self.play.md_bg_color = style.blue_violet
        else:
            self.wave_sound.play_audio()
            self.play.icon = "pause"
            self.play.md_bg_color = style.dark_sky_blue

    def press_button_back(self, _: typing.Any) -> None:
        self.wave_sound.sound_changed()

    def press_button_eraser(self, _: typing.Any) -> None:
        if self.waveform_graph.is_eraser_mode():
            # Eraser Mode -> Draw Mode
            self.waveform_graph.set_draw_mode()
            self.eraser_mode.icon = "eraser"
            self.eraser_mode.md_bg_color = style.blue_violet
        else:
            # Draw Mode -> Eraser Mode
            self.waveform_graph.set_eraser_mode()
            self.eraser_mode.icon = "pen"
            self.eraser_mode.md_bg_color = style.dark_sky_blue

    def press_button_clear(self, _: typing.Any) -> None:
        self.waveform_graph.clear_selected_points()
        self.update_waveform()

    def press_button_add(self, _: typing.Any) -> None:
        if self.num_power_spectrums < self.max_harmonics:
            self.num_power_spectrums += 1
            button = self.create_button(self.num_power_spectrums)
            button.root_wave = self
            self.power_buttons.append(button)
            self.ids.power_spectrum_buttons.add_widget(button)
            self.update_display_power_spectrum(self.num_power_spectrums - 1)
            self.harmonic_list[self.current_harmonic_index] = [self.mean.max // 2, 1, 50, 1, self.decay_function.text]
            self.update_sliders()
            self.update_power_spectrum()

    def press_button_all_power_spectrum(self, _: typing.Any) -> None:
        for slider in self.power_spectrum_sliders:
            slider.disabled = True
        self.power_buttons[self.current_harmonic_index].md_bg_color = self.unselected_button_color
        self.all_power_spectrums.md_bg_color = self.selected_button_color
        self.power_plot.points = self.sound_model.get_sum_all_power_spectrum_histogram()
        self.power_spectrum_graph.ymax = max(int(max(self.power_plot.points, key=lambda x: x[1])[1]), 1)

    def update_display_power_spectrum(self, harmonic_index: int):
        for slider in self.power_spectrum_sliders:
            slider.disabled = False
        self.all_power_spectrums.md_bg_color = self.unselected_button_color
        self.power_buttons[self.current_harmonic_index].md_bg_color = self.unselected_button_color
        self.power_buttons[harmonic_index].md_bg_color = self.selected_button_color

        self.harmonic_list[self.current_harmonic_index] = [self.mean.value, self.sd.value,
                                                           int(self.harmonic_samples.value),
                                                           int(self.num_harmonics.value),
                                                           self.decay_function.text]
        self.current_harmonic_index = harmonic_index
        self.update_sliders()

    def update_sliders(self):
        self.change_power_spectrum = False
        mean, sd, harmonic_samples, num_harmonics, decay_function = self.harmonic_list[self.current_harmonic_index]
        self.mean.value, self.sd.value, self.harmonic_samples.value = mean, sd, harmonic_samples
        self.num_harmonics.value, self.decay_function.text = num_harmonics, decay_function
        self.change_power_spectrum = True

    def press_button_display_power_spectrum(self, button: MDRectangleFlatButton):
        self.update_display_power_spectrum(int(button.text) - 1)
        self.update_power_spectrum_graph()

    def set_double_tap(self, _button, touch):
        self.double_tap = False
        if touch.is_double_tap:
            self.double_tap = True

    def create_button(self, button_num: int) -> MDRectangleFlatButton:
        return MDRectangleFlatButton(
            text=str(button_num),
            size_hint=(0.1, 1),
            md_bg_color=self.selected_button_color,
            on_press=self.press_button_display_power_spectrum,
            on_touch_down=self.set_double_tap,
            on_release=self.remove_power_spectrum,
            text_color="white",
            line_color=(0, 0, 0, 0),
        )

    def remove_power_spectrum(self, _):
        if not self.double_tap or len(self.power_buttons) == 1:
            return

        button = self.power_buttons[self.current_harmonic_index]
        self.power_buttons.remove(button)
        self.ids.power_spectrum_buttons.remove_widget(button)

        for i in range(self.current_harmonic_index, len(self.power_buttons)):
            self.power_buttons[i].text = f"{i + 1}"
            self.harmonic_list[i] = self.harmonic_list[i + 1]

        # zero fill end of harmonic list to account for removal
        self.harmonic_list[self.num_power_spectrums - 1] = [0, 0, 0, 1, "1 / x"]

        self.sound_model.remove_power_spectrum(self.current_harmonic_index, self.num_power_spectrums)

        self.num_power_spectrums -= 1
        # update current index selection
        self.current_harmonic_index -= 1 if self.current_harmonic_index == len(self.power_buttons) else 0

        self.power_buttons[self.current_harmonic_index].md_bg_color = self.selected_button_color

        self.update_sliders()
        self.update_power_spectrum_graph()
        self.update_waveform()


class WaveApp(MDApp):
    def build(self) -> RootWave:
        return RootWave()
