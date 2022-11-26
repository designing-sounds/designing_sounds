from threading import Thread
from pygame import midi


class Instrument:
    """
    Abstract instrument class.
    """

    def __init__(self):
        pass

    def begin(self):
        NotImplementedError()

    def shutdown(self):
        NotImplementedError()


class PianoMIDI(Instrument):
    """
    Piano connected through midi signal
    """

    def __init__(self):
        super().__init__()
        # -- INITIALIZATION --
        midi.init()
        default_id = midi.get_default_input_id()
        self.midi_input = midi.Input(device_id=default_id)
        self.running = False
        self.thread = None

    def begin(self) -> bool:  # Returns current running start after operation
        if not self.running:  # Not running so should start
            self.running = True
            self.thread = Thread(target=self._run_synth)
            self.thread.start()
            return True

        # Otherwise Already running so stop
        self.running = False
        self.thread.join()  # Thread should end soon so wait for it
        return  False

    def _run_synth(self):
        # -- RUN THE SYNTH --
        while self.midi_input.poll():  # Clear anything from the buffer while turned off
            self.midi_input.read(16)

        print("Starting synth...")
        notes_dict = {}
        while self.running:
            if self.midi_input.poll():
                # Add or remove notes from notes_dict
                for event in self.midi_input.read(num_events=16):
                    (status, note, _vel, _), _ = event
                    if status == 0x80 and note in notes_dict:  # Stop Note
                        pass
                    elif status == 0x90 and note not in notes_dict:  # Start Note
                        freq = midi.midi_to_frequency(note)
                        print(freq)

    def shutdown(self) -> bool:
        if self.running:
            self.running = False
            self.thread.join()
        self.midi_input.close()
        print("Stopping synth...")
        return False
