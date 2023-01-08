# Designing Sounds

Let's create a tool that allows you to draw sound waveforms. Using some signal processing and simple machine learning, we can turn points on a short bit of waveform into a continuous sound with similar properties.

The project has two main challenges:
1. Back-end: Implementing the signal processing that actually generates the sounds.
2. Front-end: Creating user-interfaces that allow different aspects of the sound to be modified.

The core deliverable will be to implement a simple Linear Regression method to interpolate the points that the user draws on the waveform, and turn them into a full sound signal that can last indefinitely. Various additional UI components will be used to e.g. 1) visualise the frequency spectrum, 2) automatically adjust the assumed frequency spectrum to the specified points.

The front-end should display the waveform, and communicate with the back-end to generate and play the sound. There are tonnes of way to manipulate the waveform, so there is lots of room to be creative. During this project, I encourage using an "agile" workflow, with "responding to change" being the golden rule. On top of the core deliverable, there is lots of room for being creative with designing sound effects and user interfaces for them. Figuring out which direction to go in depends on experimenting with what you will have created.

To see a demo of the maths behind creating a waveform, see this link: https://nbviewer.org/github/markvdw/gp-audio/blob/main/gp-audio.ipynb. This samples a waveform from a "prior" on functions using the Random Fourier Feature technique, which will be the backbone of the project. It is crucial to get the signal processing part right. Without this, the whole project falls apart. Having someone on the team who has taken computational techniques would be helpful with this (someone needs to know what a Fourier Transform is). The back-end designers *must* also demonstrate their work in Jupyter notebooks before integrating it into the UI, to ensure that the basics work.

## Building & Installing

### Mac OS X Package
For Mac OS X we have a pre-built package that you can download and install.
Head to the [Releases](https://github.com/designing-sounds/designing_sounds/releases) page and download the latest version (sounds.pkg)
Once downloaded click to open and an installer should launch and once complete the appliation will appear in your Applications folder.

### Other Platforms

At this time we are currently not providing explicit packages for other operating systems. However, all code we have developed is cross-platform.
And as it is an open source project if you would like to run it then you can download the full source code. By either cloning the repo or downloading a zip version of the code in the [Releases](https://github.com/designing-sounds/designing_sounds/releases) page.
Create a virtual environment based on the packages in the requirements.txt file.
And then run the main application as below:

```shell
python3 ./main.py
```

### Libraries Used
Kivy, numpy, pytest, pytest-cov, kivy_garden.graph, PyAudio, pyinstaller, kivymd, Cython, PyQt3D, pygame, scipy

### References
```
[1] Wilson JT, Borovitskiy V, Terenin A, Mostowsky P, Deisenroth MP. Efficiently Sampling Functions
from Gaussian Process Posteriors. 2020. Available from: https://arxiv.org/abs/2002.09309.

[2] Tompkins A, Ramos F. Fourier Feature Approximations for Periodic Kernels in Time-Series Modelling.
Proceedings of the AAAI Conference on Artificial Intelligence. 2018 Apr;32(1). Available from: https:
//ojs.aaai.org/index.php/AAAI/article/view/11696.

[3] Rahimi A, Recht B. Random Features for Large-Scale Kernel Machines. In: Platt J, Koller D,
Singer Y, Roweis S, editors. Advances in Neural Information Processing Systems. vol. 20. Cur-
ran Associates, Inc.; 2007. Available from: https://proceedings.neurips.cc/paper/2007/file/
013a006f03dbc5392effeb8f18fda755-Paper.pdf.

[4] Rasmussen CE, Williams CKI. Gaussian processes for machine learning. Cambridge, Mass. Mit Press;
```