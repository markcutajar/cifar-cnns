# cifar-cnn

A repo that presents various models to classify CIFAR images. 

## Setting up
A virtual environment using conda or venv is highly suggested for this project.
Python version: v3.6.3. <br>
__Do not use a newer version of python unless this is supported by the version of tensorflow you are using.__ <br>

To load dependencies just run `pip install -r requirements.txt` in the virtual environment or otherwise.

The data is needed and not setup in repository. This can be optained from this <a href="https://www.cs.toronto.edu/~kriz/cifar.html">link</a>.
The data should be extracted into a folder `data/` in the main directory.

A settings.py file needs to be setup in `config/` before started. An example of settings is provided in `config/`. Mostly this `example_settings.py` can be just renamed to `settings.py`.

## Quick Explanation

The repo is setup in a logical manner. All the models are found in `models.py` and data providers in `data_providers.py`. These are then called in the `run.py` file. 

To run the model specified in `run.py` just run `python run.py`.

The training model can be visualized by running `tensorboard --logdir=tf-log` in the terminal when in the root directory and then open `localhost:6006` in the browser.

__The cifar-cnns notebook has tests, results and explanations.__

__Please note this repo is still under construction.__

