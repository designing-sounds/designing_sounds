import sys

import kivy
from kivy.config import Config
Config.set('graphics', 'multisamples', '0')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.write()

from src.wave_controller import wave
from kivy.core.window import Window


if __name__ == '__main__':
    Window.maximize()
    wave.WaveApp().run()
