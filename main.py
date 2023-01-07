from kivy.config import Config

Config.set('graphics', 'width', 1920)
Config.set('graphics', 'height', 1080)
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'maxfps', '5')

from src.wave_controller import wave
from kivy.core.window import Window

if __name__ == '__main__':
    Window.maximize()
    wave.SoundsApp().run()
