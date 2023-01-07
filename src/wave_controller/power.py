import copy
import typing

from kivy.clock import mainthread
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import Graph, LinePlot
from kivymd.uix.button import MDRectangleFlatButton
from kivymd.uix.menu import MDDropdownMenu

from src.wave_controller.instruments import PianoMIDI
from src.wave_controller.save_notes import SaveNotes, State
from src.wave_view import style
from math import floor, log


class PowerSpectrumController(BoxLayout):
    max_harmonics_per_spectrum = 5
    max_power_spectrums = 5
    num_power_spectrums = 0
    current_power_spectrum_index = 0
    max_samples_per_harmonic = 500
    yaxis_extra_padding = 1.1
    yaxis_extra_sig_figs = 2
    double_tap = False

    def __init__(self, **kwargs):
        super(PowerSpectrumController, self).__init__(**kwargs)

        @mainthread
        def delayed() -> None:
            # Button Bindings
            self.add.bind(on_press=self.press_button_add)
            self.all_power_spectrums.bind(on_press=self.press_button_all_power_spectrum)
            self.power_spectrum_sliders = [self.periodic_sd, self.mean, self.periodic_lengthscale, self.num_harmonics,
                                           self.squared_sd, self.squared_lengthscale]
            self.num_harmonics.max = self.max_harmonics_per_spectrum
            self.power_spectrum_graph_samples = 2 * (self.mean.max * self.max_harmonics_per_spectrum + 1000)

            # Power Spectrum Graph
            border_color = [0, 0, 0, 1]
            self.power_spectrum_graph = Graph(border_color=border_color,
                                              xmin=0, xmax=self.power_spectrum_graph_samples // 2, ymin=0, ymax=20,
                                              padding=10, x_grid_label=True, y_grid_label=True, xlabel='Frequency (Hz)',
                                              ylabel='Power', x_ticks_major=self.power_spectrum_graph_samples // 20,
                                              y_ticks_major=10, y_ticks_minor=5, tick_color=(1, 0, 0, 0),
                                              label_options=dict(color=(0, 0, 0, 1)))

            plot_color = style.cyber_grape
            self.selected_button_color = style.dark_sky_blue
            self.unselected_button_color = style.blue_violet
            self.ids.power_spectrum.add_widget(self.power_spectrum_graph)
            self.sound_power_plot = LinePlot(color=style.red)
            self.power_plot = LinePlot(color=plot_color)
            self.power_spectrum_graph.add_plot(self.sound_power_plot)
            self.power_spectrum_graph.add_plot(self.power_plot)

            self.power_buttons = []

            self.initial_harmonic_values = [self.mean.value, self.periodic_sd.value, self.periodic_lengthscale.value,
                                            self.squared_sd.value, self.squared_lengthscale.value, 1]
            self.harmonic_list = [self.initial_harmonic_values] * self.max_power_spectrums
            self.press_button_add(None)
            self.change_power_spectrum = True

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
                    "text": "Squared Exponential Times Periodic Kernel",
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
                width_mult=7,
            )

        delayed()

    def load_state(self, state: State):
        self.remove_all_power_spectrums()
        self.variance.value = state.variance
        self.update_variance()
        self.harmonic_list = state.harmonic_list

        button = None
        for i in range(len(self.harmonic_list)):
            harmonic = state.harmonic_list[i]
            self.mean.value = harmonic[0]
            self.periodic_sd.value = harmonic[1]
            self.periodic_lengthscale.value = harmonic[2]
            self.squared_sd.value = harmonic[3]
            self.squared_lengthscale.value = harmonic[4]
            self.num_harmonics.value = harmonic[5]
            self.current_power_spectrum_index = i
            self.num_power_spectrums += 1
            button = self.create_button(self.num_power_spectrums)
            button.root_wave = self
            self.power_buttons.append(button)
            self.ids.power_spectrum_buttons.add_widget(button)
            button.md_bg_color = self.unselected_button_color
            self.update_power_spectrum()
            self.update_power_spectrum_graph()

        button.md_bg_color = self.selected_button_color

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

    def press_button_all_power_spectrum(self, _: typing.Any) -> None:
        for slider in self.power_spectrum_sliders:
            slider.disabled = True
        self.power_buttons[self.current_power_spectrum_index].md_bg_color = self.unselected_button_color
        self.all_power_spectrums.md_bg_color = self.selected_button_color
        self.power_plot.points, y_max = self.sound_model.get_sum_all_power_spectrums_graph(
            self.power_spectrum_graph_samples)
        self.update_power_spectrum_graph_axis(y_max)

    def press_button_display_power_spectrum(self, button: MDRectangleFlatButton) -> None:
        self.update_display_power_spectrum(int(button.text) - 1)
        self.update_power_spectrum_graph()

    def set_periodic_prior(self) -> None:
        self.squared_sd.disabled = True
        self.squared_lengthscale.disabled = True
        self.sound_model.change_kernel(1)
        self.update_power_spectrum()
        self.update_waveform()

    def set_mult_prior(self) -> None:
        self.squared_sd.disabled = False
        self.squared_lengthscale.disabled = False
        self.sound_model.change_kernel(0)
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

    def power_spectrum_from_freqs(self, freqs: [float]) -> None:
        for i in range(self.num_power_spectrums, 0, -1):
            self.double_tap = True
            self.remove_power_spectrum(None)
        self.double_tap = False
        self.sound_model.clear_all_power_spectrums()
        for i in range(0, min(self.max_harmonics_per_spectrum, len(freqs))):
            if i != 0:
                self.press_button_add(None)
            values = [min(freqs[i], self.mean.max), 2, 0.6, 0.01, 0.16, 1]
            values[0] = min(freqs[i], self.mean.max)
            self.harmonic_list[i] = values
            self.sound_model.update_power_spectrum(i, *values)
            self.update_sliders()
        self.update_waveform()
        self.update_power_spectrum()

    def update_power_spectrum_graph_axis(self, ymax):
        self.power_spectrum_graph.ymax = float(ymax * self.yaxis_extra_padding)
        y_ticks_major = (ymax * self.yaxis_extra_padding) / 5
        if y_ticks_major >= 1:
            self.power_spectrum_graph.y_ticks_major = int(y_ticks_major)
        else:
            sig_figs = int(abs(floor(log(y_ticks_major, 10))))
            self.power_spectrum_graph.y_ticks_major = round(y_ticks_major, sig_figs + self.yaxis_extra_sig_figs)

    def update_display_power_spectrum(self, harmonic_index: int) -> None:
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

    def update_sliders(self) -> None:
        self.change_power_spectrum = False
        harmonic = self.harmonic_list[self.current_power_spectrum_index]
        self.mean.value, self.periodic_sd.value, self.periodic_lengthscale.value, self.squared_sd.value = harmonic[:-2]
        self.squared_lengthscale.value, self.num_harmonics.value = harmonic[-2:]
        self.change_power_spectrum = True

    def set_double_tap(self, _button, touch) -> None:
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

    def remove_all_power_spectrums(self) -> None:
        for _ in range(len(self.power_buttons)):
            button = self.power_buttons[0]
            self.power_buttons.remove(button)
            self.ids.power_spectrum_buttons.remove_widget(button)
            self.sound_model.remove_power_spectrum(0)

        self.current_power_spectrum_index = 0
        self.num_power_spectrums = 0
        #
        # for i in range(self.current_power_spectrum_index, len(self.power_buttons)):
        #     self.power_buttons[i].text = f"{i + 1}"
        #     self.harmonic_list[i] = self.harmonic_list[i + 1]
        #
        # # zero fill end of harmonic list to account for removal
        # self.harmonic_list[self.num_power_spectrums - 1] = self.initial_harmonic_values
        #
        # self.sound_model.remove_power_spectrum(i)
        #
        # self.num_power_spectrums -= 1
        # # update current index selection
        # self.current_power_spectrum_index -= 1 if self.current_power_spectrum_index == len(self.power_buttons) else 0
        #
        # self.power_buttons[self.current_power_spectrum_index].md_bg_color = self.selected_button_color
        #
        # self.update_sliders()
        # self.update_power_spectrum()

    def remove_power_spectrum(self, _) -> None:
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

    def update_variance(self) -> None:
        self.sound_model.variance = self.variance.value
        self.sound_model.update_noise()
        self.update_waveform()

    def update_power_spectrum_graph(self) -> None:
        self.power_plot.points, ymax = self.sound_model.get_power_spectrum_graph(
            self.current_power_spectrum_index,
            self.power_spectrum_graph_samples)
        self.update_power_spectrum_graph_axis(ymax)

    def open_choose_kernel_menu(self) -> None:
        self.choose_kernel_menu.open()
