import typing

import numpy as np

from kivy.core.window import Window
from kivy.lang import Builder

from kivy.properties import (StringProperty, ObjectProperty)

from kivy_garden.graph import LinePlot
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.list import OneLineAvatarIconListItem, IRightBodyTouch
from kivymd.uix.menu import MDDropdownMenu

from src.wave_controller.instruments import PianoMIDI
from src.wave_controller.save_notes import SaveNotes, State
from src.wave_controller.instruments import PianoMIDI
from src.wave_controller.wave_graph import WaveformGraph
from src.wave_controller.wave_sound import WaveSound
from src.wave_model.wave_model import SoundModel
from src.wave_view import style
import src.wave_controller.power

Builder.load_file('src/wave_view/power.kv')
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
    ps_controller = ObjectProperty(None)
    sample_rate = 16000
    graph_sample_rate = 8000
    waveform_duration = 1
    chunk_duration = 0.1
    max_power_spectrums = 5

    def __init__(self, **kwargs: typing.Any):
        super().__init__(**kwargs)

        self.sound_model = SoundModel(self.ps_controller.max_harmonics_per_spectrum, self.max_power_spectrums)

        self.wave_sound = WaveSound(self.sample_rate, self.chunk_duration, self.sound_model)

        # Button bindings
        self.play.bind(on_press=self.press_button_play)
        self.back.bind(on_press=self.press_button_back)
        self.eraser_mode.bind(on_press=self.press_button_eraser)
        self.clear.bind(on_press=self.press_button_clear)
        self.resample.bind(on_press=self.press_button_resample)
        self.save_button.bind(on_press=self.press_save_button)
        self.clear_notes_button.bind(on_press=self.press_clear_notes_button)
        self.load_button.bind(on_press=self.press_load_button)
        self.connect_button.bind(on_press=self.press_button_connect)

        # Wave Graphs
        border_color = [0, 0, 0, 1]
        self.waveform_graph = WaveformGraph(update_waveform=self.update_waveform,
                                            update_waveform_graph=self.update_waveform_graph, size_hint=(1, 1),
                                            border_color=border_color,
                                            xmin=0, xmax=self.waveform_duration, ymin=-1.0, ymax=1.0, padding=10,
                                            draw_border=True, x_grid_label=True, y_grid_label=True, xlabel='Time',
                                            ylabel='Amplitude', precision="%.5g", x_grid=True, y_grid=True,
                                            y_ticks_major=0.25, label_options=dict(color=(0, 0, 0, 1)))

        self.ids.modulation.add_widget(self.waveform_graph)

        plot_color = style.cyber_grape

        self.wave_plot = LinePlot(color=plot_color, line_width=1)

        self.waveform_graph.add_plot(self.wave_plot)

        self.ps_controller.sound_model = self.sound_model
        self.ps_controller.max_power_spectrums = self.max_power_spectrums
        self.ps_controller.update_waveform = self.update_waveform
        self.ps_controller.waveform_graph = self.waveform_graph
        self.ps_controller.sound_changed = self.wave_sound.sound_changed
        self.ps_controller.update_waveform_graph = self.update_waveform_graph
        self.save_notes = SaveNotes()
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

        Window.bind(on_request_close=self.shutdown_audio)

    def update_waveform(self, update_noise=False) -> None:
        self.sound_model.interpolate_points(self.waveform_graph.get_selected_points(), update_noise)
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
            self.play.icon = "pause"
            self.play.md_bg_color = style.dark_sky_blue
            self.wave_sound.play_audio()

    def press_button_connect(self, _: typing.Any) -> None:
        if self.piano.begin(self.ps_controller.power_spectrum_from_freqs):  # Has successfully started
            self.connect_button.text = 'Disconnect MIDI Piano'
            self.connect_button.md_bg_color = style.dark_sky_blue
            self.save_button.disabled = False
            if len(self.save_notes.saved_notes) > 0:
                self.load_button.disabled = False
            self.save_button.md_bg_color = style.blue_violet
        else:  # Was already running so disconnected
            self.connect_button.text = 'Connect MIDI Piano'
            self.connect_button.md_bg_color = style.blue_violet

    def save_note_callback(self, freqs: [float]) -> None:
        print("Notes as keys: ", self.save_notes.saved_notes.keys())
        print("Saving:", freqs)
        note = freqs[(len(freqs) - 1)]
        self.save_state(note)
        self.save_button.md_bg_color = style.blue_violet
        self.load_button.disabled = False
        self.clear_notes_button.disabled = False
        self.save_notes.saving = False
        print("Notes as keys: ", self.save_notes.saved_notes.keys())
        self.piano.begin(self.save_note_callback)

    def load_notes_callback(self, freqs: [float]) -> None:
        print("Loading:", freqs)
        note = freqs[(len(freqs) - 1)]
        print("Notes as keys: ", self.save_notes.saved_notes.keys())
        if note in self.save_notes.saved_notes.keys() and len(freqs) == 1:
            state = self.save_notes.saved_notes[note]
            self.load_state(state)

    def press_save_button(self, _: typing.Any) -> None:
        if self.connect_button.text == 'Disconnect MIDI Piano':
            self.press_button_connect(None)
        self.save_button.md_bg_color = style.dark_sky_blue
        self.save_notes.saving = True
        self.piano.begin(self.save_note_callback)

    def save_state(self, note):
        self.save_notes.save(note, self.ps_controller.harmonic_list,
                             [self.ps_controller.initial_harmonic_values] * self.ps_controller.max_power_spectrums,
                             self.ps_controller.current_power_spectrum_index, self.waveform_graph.get_selected_points(),
                             self.ps_controller.variance.value)

    def press_load_button(self, _: typing.Any) -> None:
        if self.connect_button.text == 'Disconnect MIDI Piano':
            self.press_button_connect(None)
        if self.save_notes.loading:
            self.piano.begin(self.load_notes_callback)
            self.save_notes.loading = False
            self.load_button.md_bg_color = style.blue_violet
        else:
            self.save_notes.loading = True
            self.load_button.md_bg_color = style.dark_sky_blue
            self.piano.begin(self.load_notes_callback)

    def press_clear_notes_button(self, _):
        self.load_button.disabled = True
        self.clear_notes_button.disabled = True
        self.save_notes.clear_saved_notes()

    def load_state(self, state: State):
        self.ps_controller.load_state(state)
        self.waveform_graph.get_preset_points_from_y(state.points)

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

    def press_button_resample(self, _: typing.Any) -> None:
        self.sound_model.update_prior()
        self.update_waveform()

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
        self.sound_model.interpolate_points(
            self.waveform_graph.get_preset_points(waves[x], num_points, waves[x] == square_wave, waves[x] == sawtooth_wave))
        self.ps_controller.update_power_spectrum()
        self.wave_sound.sound_changed()

    def shutdown_audio(self, _) -> bool:
        self.wave_sound.shutdown()
        self.piano.shutdown()
        return False

    def open_choose_wave_menu(self) -> None:
        self.choose_wave_menu.open()


class SoundsApp(MDApp):
    def build(self) -> RootWave:
        self.icon = 'media/icon.png'
        return RootWave()
