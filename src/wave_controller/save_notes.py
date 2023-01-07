import copy


class SaveNotes:
    def __init__(self):
        self.saved_notes = dict()

    def save(self, note, harmonic_list, harmonic_dimensions, current_index):
        state = State(self.copy_harmonic_list(harmonic_list, harmonic_dimensions), current_index)
        self.saved_notes[note] = state

    def clear_saved_notes(self):
        self.saved_notes = dict()

    def copy_harmonic_list(self, harmonic_list, dimensions):
        copy_harmonic_list = dimensions
        for (i, harmonic) in enumerate(harmonic_list):
            copy_harmonic_list[i] = harmonic
            return copy_harmonic_list

    def save_power_spectrum_sliders(self, sliders) -> None:
        # for slider in sliders:
        #     self.power_spectrum_sliders.append(slider)
        pass

    def save_power_spectrum_graph(self, graph) -> None:
        pass

    def save_sound_power_plot(self, plot) -> None:
        pass

    def save_power_plot(self, plot) -> None:
        pass

    def save_power_buttons(self, buttons) -> None:
        # for button in buttons:
        #     self.power_buttons.append(copy.copy(button))
        pass


class State:
    def __init__(self, harmonic_list, index):
        self.harmonic_list = harmonic_list
        self.index = index
        self.saving = False
        self.loading = False
        self.harmonic_list = None
        self.power_spectrum_sliders = None
        self.power_spectrum_graph = None
        self.sound_power_plot = None
        self.power_plot = None
        self.power_buttons = None
