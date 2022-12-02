from kivy.config import Config

from src.wave_controller import wave
from kivy.core.window import Window

Config.set('graphics', 'width', 1920)
Config.set('graphics', 'height', 1080)
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.write()

if __name__ == '__main__':
    Window.maximize()
    wave.WaveApp().run()
