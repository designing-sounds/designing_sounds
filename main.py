from kivy.config import Config

from src.wave_controller import wave
from kivy.core.window import Window

Window.maximize()
max_size = Window.system_size
Window.size = (max_size[0], max_size[1])

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.write()

if __name__ == '__main__':
    Window.maximize()
    wave.WaveApp().run()
