from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
import simpleaudio as sa
import soundDump


class RootWidget(BoxLayout):
    def __init__(self, **kwargs):
        super(RootWidget, self).__init__(**kwargs)
        btn1 = Button(text='Play')
        btn1.bind(on_press=self.callback)

        self.add_widget(btn1)

    @staticmethod
    def callback(self):
        play_obj = sa.play_buffer(soundDump.soundDump, 2, 2, 44100)


class TestApp(App):
    def build(self):
        return RootWidget()


if __name__ == '__main__':
    TestApp().run()