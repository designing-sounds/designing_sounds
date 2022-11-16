import typing

import numpy as np
from kivy.lang import Builder
from kivy_garden.graph import LinePlot, Graph, BarPlot
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRectangleFlatButton

from src.wave_controller.wave_graph import WaveformGraph
from src.wave_controller.wave_sound import WaveSound
from src.wave_model.wave_model import PowerSpectrum
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

    def __init__(self, **kwargs: typing.Any):
        super().__init__(**kwargs)
        self.max_samples_per_harmonic = int(self.harmonic_samples.max)

        self.do_not_change_waveform = False
        self.sound_model = SoundModel(self.max_harmonics, self.max_samples_per_harmonic, int(self.mean.max))
        self.wave_sound = WaveSound(self.sample_rate, self.waveform_duration, self.chunk_duration, self.sound_model)

        self.play.bind(on_press=self.press_button_play)
        self.eraser_mode.bind(on_press=self.press_button_eraser)
        self.clear.bind(on_press=self.press_button_clear)
        self.add.bind(on_press=self.press_button_add)
        self.all_power_spectrums.bind(on_press=self.press_button_all_power_spectrum)
        self.power_spectrum_sliders = [self.sd, self.mean, self.harmonic_samples, self.num_harmonics]

        border_color = [0, 0, 0, 1]
        self.waveform_graph = WaveformGraph(update=self.update_waveform,
                                            size_hint=(1, 1),
                                            border_color=border_color,
                                            xmin=0, xmax=self.waveform_duration,
                                            ymin=-1.0, ymax=1.0,
                                            padding=10,
                                            draw_border=True,
                                            x_grid_label=True, y_grid_label=True,
                                            xlabel='Time', ylabel='Amplitude',
                                            precision="%.5g",
                                            x_grid=True, y_grid=True, y_ticks_major=0.25,
                                            label_options=dict(color=(0, 0, 0, 1)))
        self.power_spectrum_graph = Graph(border_color=border_color,
                                          xmin=0, xmax=self.mean.max,
                                          ymin=0, ymax=20,
                                          draw_border=True)

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
        self.harmonic_list = np.zeros((self.max_harmonics, 4))
        self.press_button_add(None)
        self.double_tap = False

    def update_power_spectrum(self, mean: float, sd: float, num_samples: float, num_harmonics: int) -> None:
        for slider in self.power_spectrum_sliders:
            slider.disabled = False
        if not self.do_not_change_waveform:
            self.sound_model.update_power_spectrum(self.current_harmonic_index, int(mean), sd, int(num_samples),
                                                   int(num_harmonics))
            self.update_waveform()
        self.power_plot.points = self.sound_model.get_power_spectrum_histogram(self.current_harmonic_index,
                                                                               self.power_spectrum_graph_samples)
        self.power_spectrum_graph.ymax = max(int(max(self.power_plot.points, key=lambda x: x[1])[1]), 1)

    def update_waveform(self) -> None:
        inputted_points = self.waveform_graph.get_selected_points()
        self.sound_model.interpolate_points(inputted_points)
        self.wave_sound.sound_changed()
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

    def press_button_eraser(self, _: typing.Any) -> None:
        if not self.waveform_graph.eraser_mode:
            self.waveform_graph.eraser_mode = True
            self.eraser_mode.icon = "pen"
            self.eraser_mode.md_bg_color = style.dark_sky_blue
        else:
            self.waveform_graph.eraser_mode = False
            self.eraser_mode.icon = "eraser"
            self.eraser_mode.md_bg_color = style.blue_violet
    def press_button_clear(self, _: typing.Any) -> None:
        self.waveform_graph.clear_selected_points()
        self.update_waveform()

    def press_button_add(self, _: typing.Any) -> None:
        if self.num_power_spectrums < self.max_harmonics:
            self.add_power_spectrum_button()
            self.update_power_spectrum(self.mean.value, self.sd.value, self.harmonic_samples.value,
                                       self.num_harmonics.value)

    def press_button_all_power_spectrum(self, _: typing.Any) -> None:
        for slider in self.power_spectrum_sliders:
            slider.disabled = True
        self.power_buttons[self.current_harmonic_index].md_bg_color = self.unselected_button_color
        self.all_power_spectrums.md_bg_color = self.selected_button_color
        self.power_plot.points = self.sound_model.get_sum_all_power_spectrum_histogram()
        self.power_spectrum_graph.ymax = max(int(max(self.power_plot.points, key=lambda x: x[1])[1]), 1)

    def update_display_power_spectrum(self, harmonic_index: int, change_harmonic: bool):
        self.change_selected_power_spectrum_button(harmonic_index)
        self.harmonic_list[self.current_harmonic_index] = np.array(
            [self.mean.value, self.sd.value, self.harmonic_samples.value, self.num_harmonics.value])
        self.current_harmonic_index = harmonic_index
        mean, sd, harmonic_samples, num_harmonics = self.harmonic_list[harmonic_index]
        self.do_not_change_waveform = change_harmonic
        self.mean.value = int(mean)
        self.sd.value = float(sd)
        self.harmonic_samples.value = int(harmonic_samples)
        self.num_harmonics.value = int(num_harmonics)
        self.update_power_spectrum(mean, sd, harmonic_samples, num_harmonics)
        self.do_not_change_waveform = False
        # Changing mean, sd and harmonic_samples will automatically call self.update_power_spectrum

    def press_button_display_power_spectrum(self, button: MDRectangleFlatButton):
        self.update_display_power_spectrum(int(button.text) - 1, True)

    def add_power_spectrum_button(self) -> None:
        self.num_power_spectrums += 1
        button = self.create_button(self.num_power_spectrums)
        button.root_wave = self
        self.power_buttons.append(button)
        self.ids.power_spectrum_buttons.add_widget(button)
        self.harmonic_list[self.num_power_spectrums - 1] = np.array(
            [self.mean.max // 2, 1, self.harmonic_samples.max // 2, 1])
        self.update_display_power_spectrum(self.num_power_spectrums - 1, False)

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

    def change_selected_power_spectrum_button(self, new_selection: int):
        self.all_power_spectrums.md_bg_color = self.unselected_button_color
        self.power_buttons[self.current_harmonic_index].md_bg_color = self.unselected_button_color
        self.power_buttons[new_selection].md_bg_color = self.selected_button_color

    def remove_power_spectrum(self, button: MDRectangleFlatButton):
        if not self.double_tap or len(self.power_buttons) == 1:
            return

        start_removal = self.current_harmonic_index + 1
        end_removal = len(self.power_buttons)

        # Create new buttons
        for i in range(start_removal, end_removal):
            new_button = self.create_button(i)
            self.power_buttons.append(new_button)
            self.ids.power_spectrum_buttons.add_widget(new_button)
            self.power_buttons[len(self.power_buttons) - 1].md_bg_color = self.unselected_button_color

        # Remove old buttons
        for i in range(start_removal - 1, end_removal):
            button = self.power_buttons[self.current_harmonic_index]
            self.power_buttons.remove(button)
            self.ids.power_spectrum_buttons.remove_widget(button)

        # shift harmonic list values
        for i in range(start_removal, end_removal):
            self.harmonic_list[i - 1] = self.harmonic_list[i]

        # zero fill end of harmonic list to account for removal
        self.harmonic_list[end_removal - 1] = np.array((0, 0, 0, 0))
        self.num_power_spectrums -= 1

        # update power spectrum values
        self.sound_model.power_spectrum = PowerSpectrum(self.max_harmonics, self.max_samples_per_harmonic)
        for i in range(self.max_harmonics):
            (mean, sd, num_samples, num_harmonics) = self.harmonic_list[i]
            self.sound_model.power_spectrum.update_harmonic(i, mean, sd, int(num_samples), int(num_harmonics))

        # update current index selection
        if self.current_harmonic_index == len(self.power_buttons):
            self.current_harmonic_index -= 1
        self.power_buttons[self.current_harmonic_index].md_bg_color = self.selected_button_color

        self.do_not_change_waveform = False

        (mean, sd, num_samples, num_harmonics) = self.harmonic_list[self.current_harmonic_index]
        (self.mean.value, self.sd.value, self.harmonic_samples.value, self.num_harmonics.value) = (
            int(mean), float(sd), int(num_samples), int(num_harmonics))

        self.update_power_spectrum(mean, sd, num_samples, num_harmonics)


class WaveApp(MDApp):
    def build(self) -> RootWave:
        return RootWave()
