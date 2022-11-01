import sys

import kivy
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.write()

from src.wave_view import wave
from kivy.core.window import Window


if __name__ == '__main__':
    kivy.resources.resource_add_path(sys._MEIPASS)
    Window.maximize()
    wave.WaveApp().run()
