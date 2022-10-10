# Designing Sounds

Let's create a tool that allows you to draw sound waveforms. Using some signal processing and simple machine learning, we can turn points on a short bit of waveform into a continuous sound with similar properties.

The project has two main challenges:
1. Back-end: Implementing the signal processing that actually generates the sounds.
2. Front-end: Creating user-interfaces that allow different aspects of the sound to be modified.

The core deliverable will be to implement a simple Linear Regression method to interpolate the points that the user draws on the waveform, and turn them into a full sound signal that can last indefinitely. Various additional UI components will be used to e.g. 1) visualise the frequency spectrum, 2) automatically adjust the assumed frequency spectrum to the specified points.

The front-end should display the waveform, and communicate with the back-end to generate and play the sound. There are tonnes of way to manipulate the waveform, so there is lots of room to be creative. During this project, I encourage using an "agile" workflow, with "responding to change" being the golden rule. On top of the core deliverable, there is lots of room for being creative with designing sound effects and user interfaces for them. Figuring out which direction to go in depends on experimenting with what you will have created.

To see a demo of the maths behind creating a waveform, see this link: https://nbviewer.org/github/markvdw/gp-audio/blob/main/gp-audio.ipynb. This samples a waveform from a "prior" on functions using the Random Fourier Feature technique, which will be the backbone of the project. It is crucial to get the signal processing part right. Without this, the whole project falls apart. Having someone on the team who has taken computational techniques would be helpful with this (someone needs to know what a Fourier Transform is). The back-end designers *must* also demonstrate their work in Jupyter notebooks before integrating it into the UI, to ensure that the basics work.

## Available Scripts

In the project directory, you can run:

First install:

### `npm install`

then run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
