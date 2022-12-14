from kivy.config import Config

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'maxfps', '10')

from src.wave_controller import wave
from kivy.core.window import Window

if __name__ == '__main__':
    Window.maximize()
    wave.SoundsApp().run()
