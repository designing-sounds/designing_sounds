import typing

import numpy as np
from kivy.core.window import Window
from kivy.lang import Builder

from kivy.properties import StringProperty
from kivy_garden.graph import LinePlot, Graph
from kivymd.app import MDApp
from kivymd.toast import toast
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRectangleFlatButton
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.list import OneLineAvatarIconListItem, IRightBodyTouch
from kivymd.uix.menu import MDDropdownMenu
from scipy.io import wavfile

from src.wave_controller.instruments import PianoMIDI
from src.wave_controller.wave_graph import WaveformGraph
from src.wave_controller.wave_sound import WaveSound
from src.wave_model.wave_model import SoundModel
from src.wave_view import style

Builder.load_file('src/wave_view/wave.kv')

SINE_WAVE = 0
SQUARE_WAVE = 1
TRIANGLE_WAVE = 2
SAWTOOTH_WAVE = 3


class RightContentCls(IRightBodyTouch, MDBoxLayout):
    icon = StringProperty()
    text = StringProperty()


class Item(OneLineAvatarIconListItem):
    left_icon = StringProperty()
    right_icon = StringProperty()
    right_text = StringProperty()


class RootWave(MDBoxLayout):
    sample_rate = 16000
    graph_sample_rate = 2500
    waveform_duration = 1
    chunk_duration = 0.01

    max_harmonics = 5
    max_power_spectrums = 5
    num_power_spectrums = 0
    current_power_spectrum_index = 0
    max_samples_per_harmonic = 500

    def __init__(self, **kwargs: typing.Any):
        super().__init__(**kwargs)
        self.is_showing = False
        self.file_manager = None
        self.loaded_file = None
        self.power_spectrum_graph_samples = 2 * (self.mean.max * self.max_harmonics + 1000)
        self.num_harmonics.max = self.max_harmonics
        self.max_harmonics = self.num_harmonics.max
        self.sound_model = SoundModel(self.max_power_spectrums, int(self.mean.max), self.max_harmonics)
        self.wave_sound = WaveSound(self.sample_rate, self.waveform_duration, self.chunk_duration, self.sound_model)

        # Button bindings
        self.play.bind(on_press=self.press_button_play)
        self.back.bind(on_press=self.press_button_back)
        self.eraser_mode.bind(on_press=self.press_button_eraser)
        self.clear.bind(on_press=self.press_button_clear)
        self.resample.bind(on_press=self.press_button_resample)
        self.add.bind(on_press=self.press_button_add)
        self.show_loaded.bind(on_press=self.press_button_show_loaded_sound)

        self.all_power_spectrums.bind(on_press=self.press_button_all_power_spectrum)
        self.connect_button.bind(on_press=self.press_button_connect)
        self.power_spectrum_sliders = [self.periodic_sd, self.mean, self.periodic_lengthscale, self.num_harmonics,
                                       self.squared_sd, self.squared_lengthscale]

        # Wave Graphs
        border_color = [0, 0, 0, 1]
        self.waveform_graph = WaveformGraph(update_waveform=self.update_waveform,
                                            update_waveform_graph=self.update_waveform_graph, size_hint=(1, 1),
                                            border_color=border_color,
                                            xmin=0, xmax=self.waveform_duration, ymin=-1.0, ymax=1.0, padding=10,
                                            draw_border=True, x_grid_label=True, y_grid_label=True, xlabel='Time',
                                            ylabel='Amplitude', precision="%.5g", x_grid=True, y_grid=True,
                                            y_ticks_major=0.25, label_options=dict(color=(0, 0, 0, 1)))
        self.power_spectrum_graph = Graph(border_color=border_color,
                                          xmin=0, xmax=self.power_spectrum_graph_samples // 2, ymin=0, ymax=20,
                                          padding=10, x_grid_label=True, y_grid_label=True, xlabel='Frequency (Hz)',
                                          ylabel='Samples', x_ticks_major=self.power_spectrum_graph_samples // 20,
                                          y_ticks_major=10, y_ticks_minor=5, tick_color=(1, 0, 0, 0),
                                          label_options=dict(color=(0, 0, 0, 1)))

        self.ids.modulation.add_widget(self.waveform_graph)
        self.ids.power_spectrum.add_widget(self.power_spectrum_graph)

        plot_color = style.cyber_grape

        self.wave_plot = LinePlot(color=plot_color, line_width=1)
        self.power_plot = LinePlot(color=plot_color)
        self.load_sound_plot = LinePlot(color=style.red, line_width=1)
        self.sound_power_plot = LinePlot(color=style.red)

        self.waveform_graph.add_plot(self.wave_plot)
        self.power_spectrum_graph.add_plot(self.power_plot)
        self.waveform_graph.add_plot(self.load_sound_plot)
        self.power_spectrum_graph.add_plot(self.sound_power_plot)

        self.power_buttons = []
        self.selected_button_color = style.dark_sky_blue
        self.unselected_button_color = style.blue_violet
        self.initial_harmonic_values = [self.mean.value, self.periodic_sd.value, self.periodic_lengthscale.value,
                                        self.squared_sd.value, self.squared_lengthscale.value, 1]
        self.harmonic_list = [self.initial_harmonic_values] * self.max_power_spectrums
        self.press_button_add(None)
        self.double_tap = False
        self.change_power_spectrum = True
        self.piano = PianoMIDI()

        choose_wave_menu_items = [
            {
                "text": "Sine Wave",
                "right_text": "",
                "right_icon": "",
                "left_icon": "sine-wave",
                "viewclass": "Item",
                "on_release": lambda x=True: self.preset_waves(SINE_WAVE),
            },
            {
                "text": "Square Wave",
                "right_text": "",
                "right_icon": "",
                "left_icon": "square-wave",
                "viewclass": "Item",
                "on_release": lambda x=True: self.preset_waves(SQUARE_WAVE),
            },
            {
                "text": "Triangle Wave",
                "right_text": "",
                "right_icon": "",
                "left_icon": "triangle-wave",
                "viewclass": "Item",
                "on_release": lambda x=True: self.preset_waves(TRIANGLE_WAVE),
            },
            {
                "text": "Sawtooth Wave",
                "right_text": "",
                "right_icon": "",
                "left_icon": "sawtooth-wave",
                "viewclass": "Item",
                "on_release": lambda x=True: self.preset_waves(SAWTOOTH_WAVE),
            }
        ]
        self.choose_wave_menu = MDDropdownMenu(
            caller=self.preset,
            items=choose_wave_menu_items,
            width_mult=4,
        )
        choose_kernel_menu_items = [
            {
                "text": "Periodic Kernel",
                "right_text": "",
                "right_icon": "",
                "left_icon": "sine-wave",
                "viewclass": "Item",
                "on_release": lambda x=True: self.set_periodic_prior(),
            },
            {
                "text": "Gaussian Periodic Kernel",
                "right_text": "",
                "right_icon": "",
                "left_icon": "waveform",
                "viewclass": "Item",
                "on_release": lambda x=True: self.set_mult_prior(),
            }
        ]
        self.choose_kernel_menu = MDDropdownMenu(
            caller=self.kernel,
            items=choose_kernel_menu_items,
            width_mult=5,
        )

        Window.bind(on_request_close=self.shutdown_audio)

    def update_variance(self):
        self.sound_model.variance = self.variance.value
        self.sound_model.update_noise()
        self.update_waveform()

    def update_power_spectrum(self) -> None:
        if self.change_power_spectrum:
            self.sound_model.update_power_spectrum(self.current_power_spectrum_index, self.mean.value,
                                                   self.periodic_sd.value, self.periodic_lengthscale.value,
                                                   self.squared_sd.value, self.squared_lengthscale.value,
                                                   int(self.num_harmonics.value))
            self.update_power_spectrum_graph()
            self.update_waveform()
            self.waveform_graph.set_period(self.mean.value)

    def power_spectrum_from_freqs(self, freqs: [float]):
        for i in range(self.num_power_spectrums, 0, -1):
            self.double_tap = True
            self.remove_power_spectrum(None)
        self.double_tap = False
        self.sound_model.clear_all_power_spectrums()
        for i in range(0, min(self.max_harmonics, len(freqs))):
            if i != 0:
                self.press_button_add(None)
            values = list(self.initial_harmonic_values)
            values[0] = min(freqs[i], self.mean.max)
            self.harmonic_list[i] = values
            self.sound_model.update_power_spectrum(i, *values, 1)
            self.update_sliders()
        self.update_waveform()
        self.update_power_spectrum()

    def update_power_spectrum_graph(self):
        self.power_plot.points, ymax = self.sound_model.get_power_spectrum_histogram(
            self.current_power_spectrum_index,
            self.power_spectrum_graph_samples)
        self.power_spectrum_graph.ymax = float(ymax)
        self.power_spectrum_graph.y_ticks_major = max(int(self.power_spectrum_graph.ymax / 5), 1)

    def update_waveform(self) -> None:
        self.sound_model.interpolate_points(self.waveform_graph.get_selected_points())
        self.wave_sound.sound_changed()
        self.update_waveform_graph()

    def update_loaded_sound_graph(self) -> None:
        x_min = self.waveform_graph.xmin
        x_max = self.waveform_graph.xmax
        sample_rate, data = self.loaded_file
        start_index = int(sample_rate * x_min)
        finish_index = int(sample_rate * x_max)
        self.load_sound_plot.points = list(
            zip(np.linspace(x_min, x_max, finish_index - start_index), data[start_index:finish_index]))

    def update_waveform_graph(self) -> None:
        x_min = self.waveform_graph.xmin
        x_max = self.waveform_graph.xmax
        points = self.sound_model.model_sound(self.graph_sample_rate / (x_max - x_min), x_max - x_min, x_min)
        self.wave_plot.points = list(zip(np.linspace(x_min, x_max, points.size), points))
        if self.loaded_file and self.is_showing:
            self.update_loaded_sound_graph()

    def press_button_play(self, _: typing.Any) -> None:
        if self.wave_sound.is_playing():
            self.wave_sound.pause_audio()
            self.play.icon = "play"
            self.play.md_bg_color = style.blue_violet
        else:
            self.wave_sound.play_audio()
            self.play.icon = "pause"
            self.play.md_bg_color = style.dark_sky_blue

    def press_button_connect(self, _: typing.Any) -> None:
        if self.piano.begin(self.power_spectrum_from_freqs):  # Has successfully started
            self.connect_button.text = 'Disconnect MIDI Piano Power Spectrum'
            self.connect_button.md_bg_color = style.dark_sky_blue
        else:  # Was already running so disconnected
            self.connect_button.text = 'Connect MIDI Piano Power Spectrum'
            self.connect_button.md_bg_color = style.blue_violet

    def press_button_show_loaded_sound(self, _: typing.Any) -> None:
        if self.loaded_file:
            if self.is_showing:
                # Hide the graphs
                self.is_showing = False
                self.show_loaded.icon = "cellphone-sound"
                self.show_loaded.text = "Show Loaded Sound"
                self.load_sound_plot.points = []
                self.sound_power_plot.points = []
                self.show_loaded.md_bg_color = style.blue_violet
            else:
                self.is_showing = True
                self.show_loaded.icon = "file-hidden"
                self.show_loaded.text = "Hide Loaded Sound"
                _, data = self.loaded_file
                self.update_loaded_sound_graph()
                self.sound_power_plot.points = self.sound_model.get_power_spectrum(data)
                self.show_loaded.md_bg_color = style.dark_sky_blue

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
        self.sound_model.update_prior()
        self.update_waveform()

    def press_button_resample(self, _: typing.Any) -> None:
        self.sound_model.update_prior()
        self.update_waveform()

    def press_button_add(self, _: typing.Any) -> None:
        if self.num_power_spectrums < self.max_power_spectrums:
            self.num_power_spectrums += 1
            button = self.create_button(self.num_power_spectrums)
            button.root_wave = self
            self.power_buttons.append(button)
            self.ids.power_spectrum_buttons.add_widget(button)
            self.update_display_power_spectrum(self.num_power_spectrums - 1)
            self.harmonic_list[self.current_power_spectrum_index] = self.initial_harmonic_values
            self.update_sliders()
            self.update_power_spectrum()

    def preset_waves(self, x: int):
        num_points = 100

        def sin_wave(x, period):
            amp_scale = 0.75
            scale = (2 * np.pi)
            return amp_scale * np.sin((scale / period) * x)

        def square_wave(x, period):
            square_scale = 0.75
            return -square_scale if x < (period / 2) else square_scale

        def triangle_wave(x, period):
            scale = 3
            slope = (scale / period)
            scale_factor = scale / 4
            if 0 <= x < period / 4:
                return slope * x
            if period / 4 <= x < 3 * period / 4:
                return scale_factor * 2 - slope * x
            return slope * x - scale_factor * 4

        def sawtooth_wave(x, period):
            return 3 / 2 / period * x - 3 / 4

        waves = [sin_wave, square_wave, triangle_wave, sawtooth_wave]
        self.sound_model.interpolate_points(self.waveform_graph.get_preset_points(waves[x], num_points))
        self.wave_sound.sound_changed()
        self.update_power_spectrum()

    def set_periodic_prior(self):
        self.squared_sd.disabled = True
        self.squared_lengthscale.disabled = True
        self.sound_model.set_periodic_prior()
        self.update_waveform()

    def set_mult_prior(self):
        self.squared_sd.disabled = False
        self.squared_lengthscale.disabled = False
        self.sound_model.set_mult_prior()
        self.update_waveform()

    def press_button_all_power_spectrum(self, _: typing.Any) -> None:
        for slider in self.power_spectrum_sliders:
            slider.disabled = True
        self.power_buttons[self.current_power_spectrum_index].md_bg_color = self.unselected_button_color
        self.all_power_spectrums.md_bg_color = self.selected_button_color
        self.power_plot.points, y_max = self.sound_model.get_sum_all_power_spectrum_histogram(
            self.power_spectrum_graph_samples)
        self.power_spectrum_graph.ymax = max(int(y_max), 1)
        self.power_spectrum_graph.y_ticks_major = max(int(self.power_spectrum_graph.ymax / 5), 1)

    def update_display_power_spectrum(self, harmonic_index: int):
        for slider in self.power_spectrum_sliders:
            slider.disabled = False
        self.all_power_spectrums.md_bg_color = self.unselected_button_color
        self.power_buttons[self.current_power_spectrum_index].md_bg_color = self.unselected_button_color
        self.power_buttons[harmonic_index].md_bg_color = self.selected_button_color

        self.harmonic_list[self.current_power_spectrum_index] = [self.mean.value, self.periodic_sd.value,
                                                                 self.periodic_lengthscale.value,
                                                                 self.squared_sd.value, self.squared_lengthscale.value,
                                                                 self.num_harmonics.value]
        self.current_power_spectrum_index = harmonic_index
        self.update_sliders()

    def update_sliders(self):
        self.change_power_spectrum = False
        harmonic = self.harmonic_list[self.current_power_spectrum_index]
        self.mean.value, self.periodic_sd.value, self.periodic_lengthscale.value, self.squared_sd.value = harmonic[:-2]
        self.squared_lengthscale.value, self.num_harmonics.value = harmonic[-2:]
        self.change_power_spectrum = True

    def press_button_display_power_spectrum(self, button: MDRectangleFlatButton):
        self.update_display_power_spectrum(int(button.text) - 1)
        self.update_power_spectrum_graph()
        self.waveform_graph.set_period(self.mean.value)

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

        button = self.power_buttons[self.current_power_spectrum_index]
        self.power_buttons.remove(button)
        self.ids.power_spectrum_buttons.remove_widget(button)

        for i in range(self.current_power_spectrum_index, len(self.power_buttons)):
            self.power_buttons[i].text = f"{i + 1}"
            self.harmonic_list[i] = self.harmonic_list[i + 1]

        # zero fill end of harmonic list to account for removal
        self.harmonic_list[self.num_power_spectrums - 1] = self.initial_harmonic_values

        self.sound_model.remove_power_spectrum(self.current_power_spectrum_index)

        self.num_power_spectrums -= 1
        # update current index selection
        self.current_power_spectrum_index -= 1 if self.current_power_spectrum_index == len(self.power_buttons) else 0

        self.power_buttons[self.current_power_spectrum_index].md_bg_color = self.selected_button_color

        self.update_sliders()
        self.update_power_spectrum()

    def shutdown_audio(self, _) -> bool:
        self.wave_sound.shutdown()
        self.piano.shutdown()
        return False

    def open_choose_wave_menu(self) -> None:
        self.choose_wave_menu.open()

    def open_choose_kernel_menu(self) -> None:
        self.choose_kernel_menu.open()


    def file_manager_open(self) -> None:
        if not self.file_manager:
            self.file_manager = MDFileManager(
                exit_manager=self.exit_manager, select_path=self.select_path)
        self.file_manager.show('/')  # output manager to the screen

    def select_path(self, path: str) -> None:
        try:
            self.loaded_file = wavfile.read(path)
            data = np.sum(self.loaded_file[1], axis=1)
            data = data / max(data.max(), data.min(), key=abs)
            self.is_showing = True
            self.show_loaded.disabled = False
            self.loaded_file = (self.loaded_file[0], data)
            self.sound_power_plot.points = self.sound_model.get_power_spectrum(data)
            step = data.size // 500
            y = data[:self.sample_rate // 4:step]
            points = [(float(i) * step / 44100, y[i]) for i in np.arange(y.size)]
            self.sound_model.interpolate_points(self.waveform_graph.get_preset_points_from_y(points))
            self.wave_sound.sound_changed()
            self.update_power_spectrum()

            self.update_loaded_sound_graph()
            self.exit_manager()
            toast("File Loaded Successfully")
        except ValueError:
            toast("Not a valid file")

    def exit_manager(self, *_: typing.Any) -> None:
        self.file_manager.close()


class WaveApp(MDApp):
    def build(self) -> RootWave:
        return RootWave()
