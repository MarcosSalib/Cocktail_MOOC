# Introduction to Deep Learning (IN2346)
# Technical University Munich - WS 2021

## 1. Python Setup

Prerequisites:
- Unix system (Linux or MacOS)
- Python version 3.7.x
- Terminal (e.g. iTerm2 for MacOS)
- Integrated development environment (IDE) (e.g. PyCharm or Sublime Text) or text editor

For the following description, we assume that you are using Linux or MacOS and that you are familiar with working from a terminal. The exercises are implemented in Python 3.7. Note that you might be unable to install some libraries required for the assignments if your python version < 3.7. So please make sure that you install python 3.7 before proceeding.

If you are using Windows, the procedure might slightly vary and you will have to google for the details. We'll mention some of them in this document.

To avoid issues with different versions of Python and Python packages we recommend to always set up a project specific virtual environment. The most common tools for a clean management of Python environments are *pyenv*, *virtualenv* and *Anaconda*.

In this README we provide you with a short tutorial on how to use and setup a *virtuelenv* environment. To this end, install or upgrade *virtualenv*. There are several ways depending on your OS. At the end of the day, we want

`which virtualenv`

to point to the installed location.

On Ubuntu, you can use:

`apt-get install python-virtualenv`

Also, installing with pip should work (the *virtualenv* executable should be added to your search path automatically):

`pip3 install virtualenv`

Once *virtualenv* is successfully installed, go to the root directory of the i2dl repository (where this README.md is located) and execute:

`virtualenv -p python3 .venv`

Basically, this installs a sandboxed Python in the directory `.venv`. The
additional argument ensures that sandboxed packages are used even if they had
already been installed globally before.

Whenever you want to use this *virtualenv* in a shell you have to first
activate it by calling:

`source .venv/bin/activate`

To test whether your *virtualenv* activation has worked, call:

`which python`

This should now point to `.venv/bin/python`.

From now on we assume that that you have activated your virtual environment.

Installing required packages:
We have made it easy for you to get started, just call from the i2dl root directory:

`pip3 install -r requirements.txt`


The exercises are guided via Jupyter Notebooks (files ending with `*.ipynb`). In order to open a notebook dedicate a separate shell to run a Jupyter Notebook server in the i2dl root directory by executing:

`jupyter notebook`

A browser window which depicts the file structure of directory should open (we tested this with Chrome). From here you can select an exercise directory and one of its exercise notebooks!

Note:For windows, use miniconda or conda. Create an environment using the command:

`conda create --name i2dl python=3.7`

Next activate the environment using the command:

`conda activate i2dl`

Continue with installation of requirements and starting jupyter notebook as mentioned above, i.e.

`pip install -r requirements.txt` 
`jupyter notebook`


## 2. Exercise Download

The exercises will be shared on both our website and Piazza. At each time we start with a new exercise you have to unzip it into the main i2dl folder you get from the first exercise.

### The directory layout for the exercises

    i2dl_exercises
    ├── datasets                   # The datasets required for all exercises will be placed here
    ├── exercise_01                 
    ├── exercise_02                     
    ├── exercise_03                    
    ├── exercise_04
    ├── exercise_05
    ├── exercise_06
    ├── exercise_07                              
    ├── exercise_08
    ├── exercise_09
    ├── exercise_10
    ├── exercise_11
    ├── exercise_12                    
    ├── LICENSE
    └── README.md


## 3. Dataset Download

Datasets will be automatically downloaded in our jupyter notebooks. A sample directory structure for cifar10 dataset is shown below:

    i2dl_exercises
        ├── datasets                   # The datasets required for all exercises will be downloaded here
            ├── cifar10                # Dataset directory
                ├── cifar10.p          # dataset files 

## 4. Exercise Submission

Your trained models will be automatically evaluated on a test set on our server. To this end, login or register for an account at:

https://dvl.in.tum.de/teaching/submission/

Note that only students, who have registered for this class in TUM Online can register for an account. This account provides you with temporary credentials to login onto the machines at our chair.

After you have worked through an exercise, you need to execute the last cell to create a zip archive. All your trained models should be inside `models` directory in the exercise folder. 

You can login to the above website and upload your zip submission. Once uploaded you can see the all the models you submitted (if there's a blank page for a long while, refresh). Also, you can select the models for evaluation. 

You will receive an email notification with the results upon completion of the evaluation. To make it more fun, you will be able to see a leaderboard of everyone's (anonymous) scores on the login part of the submission website.

Note that you can re-evaluate your models until the deadline of the current exercise. Whereas the email contains the result of the current evaluation, the entry in the leader board always represents the best score for the respective exercise.


## 5. Acknowledgments

We want to thank the **Stanford Vision Lab** for allowing us to build these exercises on material they had previously developed.
