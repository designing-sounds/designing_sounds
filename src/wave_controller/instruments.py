from kivy.clock import Clock
from kivymd.uix.dialog import MDDialog
from pygame import midi
from pygame.midi import MidiException


class PianoMIDI():
    """
    Piano connected through midi signal
    """

    def __init__(self):
        super().__init__()
        # -- INITIALIZATION --
        self.notes_set = set()
        self.play_notes = set()
        self.changed = False
        self.midi_input = None
        self.callback_update = None
        self.running = False
        self.thread = None

    def begin(self, callback_update) -> bool:  # Returns current running start after operation
        self.callback_update = callback_update
        if not self.running:  # Not running so should start
            print(midi.get_init())
            midi.init()
            self.running = True
            if self._run_synth():
                return True
        else:
            print("Stopping synth...")

        # Otherwise Already running so stop
        self.running = False
        midi.quit()
        return False

    def _run_synth(self) -> bool:
        default_id = midi.get_default_input_id()
        if default_id == -1:  # invalid id -> no device connected
            self.dialog = MDDialog(
                title="No MIDI Piano has been detected",
                text="Click anywhere else on the screen to return"
            )
            self.dialog.open()
            return False

        try:
            self.midi_input = midi.Input(device_id=default_id)
        except MidiException:
            return False

        # -- RUN THE SYNTH --
        while self.midi_input.poll():  # Clear anything from the buffer while turned off
            self.midi_input.read(16)

        print("Starting synth...")

        Clock.schedule_interval(self.loop, 0.1)
        return True

    def loop(self, _):
        try:
            if midi.get_init() & self.midi_input.poll():
                # Add or remove notes from notes_dict
                for event in self.midi_input.read(num_events=16):
                    (status, note, _vel, _), _ = event
                    if status == 0x80 and note in self.notes_set:  # Stop Note
                        self.notes_set.remove(note)
                        self.play_notes.remove(note)
                    elif status == 0x90 and note not in self.notes_set:  # Start Note
                        self.notes_set.add(note)
                        self.play_notes.add(note)
                        self.changed = True
                        # freq = midi.midi_to_frequency(note)
            if self.changed:
                freqs = list(map(midi.midi_to_frequency, self.play_notes))
                if len(freqs) > 0:
                    self.callback_update(freqs)
                self.changed = False
            return self.running
        except RuntimeError:
            return False

    def shutdown(self) -> bool:
        if self.running:
            self.running = False
            self.midi_input.close()
        print("Stopping synth...")
        return False
