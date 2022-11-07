# Designing Sounds

Let's create a tool that allows you to draw sound waveforms. Using some signal processing and simple machine learning, we can turn points on a short bit of waveform into a continuous sound with similar properties.

The project has two main challenges:
1. Back-end: Implementing the signal processing that actually generates the sounds.
2. Front-end: Creating user-interfaces that allow different aspects of the sound to be modified.

The core deliverable will be to implement a simple Linear Regression method to interpolate the points that the user draws on the waveform, and turn them into a full sound signal that can last indefinitely. Various additional UI components will be used to e.g. 1) visualise the frequency spectrum, 2) automatically adjust the assumed frequency spectrum to the specified points.

The front-end should display the waveform, and communicate with the back-end to generate and play the sound. There are tonnes of way to manipulate the waveform, so there is lots of room to be creative. During this project, I encourage using an "agile" workflow, with "responding to change" being the golden rule. On top of the core deliverable, there is lots of room for being creative with designing sound effects and user interfaces for them. Figuring out which direction to go in depends on experimenting with what you will have created.

To see a demo of the maths behind creating a waveform, see this link: https://nbviewer.org/github/markvdw/gp-audio/blob/main/gp-audio.ipynb. This samples a waveform from a "prior" on functions using the Random Fourier Feature technique, which will be the backbone of the project. It is crucial to get the signal processing part right. Without this, the whole project falls apart. Having someone on the team who has taken computational techniques would be helpful with this (someone needs to know what a Fourier Transform is). The back-end designers *must* also demonstrate their work in Jupyter notebooks before integrating it into the UI, to ensure that the basics work.

## Building & Installing

### Mac OS X
For Mac OS X we have a pre-built application that you can download and install.
Head to the [Releases](https://github.com/designing-sounds/designing_sounds/releases) page and download the latest version (sounds.dmg)
Once downloaded click to open and then click again to open the application.

#### Warning
You may get a warning such as:

- “sounds” can’t be opened because Apple cannot check it for malicious software

This is an issue that is currently being worked on to do with signing of the application. To get around this head to System Preferences > Privacy & Security and you should see a message:

- "sounds" was blocked from use because it is not from an identified developer

Click the button "Open Anyway" and the application should start  :)

### Running the application

The application is run by calling the main.py file with

```shell
python3 ./main.py
```
