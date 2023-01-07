import copy


class SaveNotes:
    def __init__(self):
        self.loading = False
        self.saved_notes = dict()

    def save(self, note, harmonic_list, harmonic_dimensions, current_index, points, variance):
        state = State(self.copy_harmonic_list(harmonic_list, harmonic_dimensions), current_index, points, variance)
        self.saved_notes[note] = state

    def clear_saved_notes(self):
        self.saved_notes = dict()

    def copy_harmonic_list(self, harmonic_list, dimensions):
        copy_harmonic_list = dimensions
        for (i, harmonic) in enumerate(harmonic_list):
            copy_harmonic_list[i] = harmonic
            return copy_harmonic_list


class State:
    def __init__(self, harmonic_list, index, points, variance):
        self.harmonic_list = harmonic_list
        self.index = index
        self.points = points
        self.variance = variance
