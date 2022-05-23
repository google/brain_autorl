# Code for RL-DARTS.

This codebase hosts the core algorithms used in the paper, as well as the DARTS + baseline policies.
For a tutorial on how they should be used, please see the associated tests.
Required packages can be installed from `requirements.txt`.


## Algorithms
* PPO-side experiments can be found in `/algorithms/ppo/run_ppo.py`, which will reproduce PPO results on Procgen.
* RainbowDQN-side experiments can be found in `/algorithms/rainbow/lp_dqn.py` which will also reproduce Rainbow experiments on Procgen.
* SAC + DM-Control experiments can be found in `/algorithms/sac/train_pisac.py`, with only the files that needed to be changed. All other files can be found in the [public repository](https://github.com/google-research/pisac).


## Policies
* The baseline IMPALA-CNN policy can be found in `/policies/base_policies.py`.
* The DARTS policies can be found in `/policies/darts_policies.py`.


In order to run DARTS-based experiments as shown in the paper, one simply needs to swap the construction of the baseline policy network with a DARTS network. For example, if the code uses:
```
network = base_policies.ImpalaCNN()
```
then one may instead use:
```
network = darts_policies.DartsStandardCNN()
```

Hyperparameter settings are outlined in the code, and further explained in the [paper submission](https://arxiv.org/abs/2106.02229).
