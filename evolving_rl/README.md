# Evolving Reinforcement Learning Algorithms


This is the codebase for [Evolving Reinforcement Learning Algorithms](https://arxiv.org/abs/2101.03958) _John D. Co-Reyes, Yingjie Miao, Daiyi Peng, Esteban Real, Sergey Levine, Quoc V. Le, Honglak Lee, Aleksandra Faust._ (ICLR Oral, 2021).

JD wrote much of the original codes, with contributions from Yingjie, Daiyi and Esteban.

# Quick Start

This [colab](https://github.com/google/brain_autorl/blob/main/evolving_rl/EvolvingRL_Demo.ipynb) contains a demo of evolving 100 algorithms on the gym CartPole environment. (Currently, the colab isn't runnable on the cloud -- we will fix this once we make brain_autorl installable via `pip`.)

Note that in the orignal paper, we used 300 distributed CPU workers for ~3 days, and evolved more than 20K algorithms on multiple RL environments. We plan to open source the distributed training codes in the future.

# Installation

The following procedure was tested on a Linux machine with Python3 installed. We assume you have `pip` installed.

We will use $HOME as a working directory example.

```
# Download the repo
> cd $HOME
> git clone https://github.com/google/brain_autorl.git

# Set up a virtual env and activate it.
> python3 -m venv test_autorl
> source test_autorl/bin/activate

# Install dependencies
(test_autorl) > export PYTHONPATH="$HOME"
(test_autorl) > cd brain_autorl/evolving_rl
(test_autorl) > python3 -m pip install -r requirements.txt

# Evolve!
(test_autorl) > python3 run_search.py
```
