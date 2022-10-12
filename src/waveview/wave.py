from kivy.app import App
from kivy.uix.widget import Widget
import simpleaudio as sa

import soundDump


class RootWave(Widget):
    def __init__(self, **kwargs):
        super(RootWave, self).__init__(**kwargs)
        self.play.bind(on_press=self.callback)

    @staticmethod
    def callback(self):
        sa.play_buffer(soundDump.soundDump, 2, 2, 44100)


class WaveApp(App):
    def build(self):
        return RootWave()
