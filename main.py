from kivy.config import Config

Config.set('graphics', 'width', 1920)
Config.set('graphics', 'height', 1080)
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'maxfps', '5')

from src.wave_controller import wave
from kivy.core.window import Window
from kivy import Logger, LOG_LEVELS

if __name__ == '__main__':
    Logger.setLevel(LOG_LEVELS["trace"])
    Window.maximize()
    Window.set_icon('media/icon.png')
    wave.SoundsApp().run()
