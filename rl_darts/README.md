# Code for RL-DARTS.

This codebase hosts the core algorithms used in the paper, as well as the DARTS + baseline policies.
For a tutorial on how they should be used, please see the associated tests.


## Algorithms
* PPO-side experiments can be found in `/algorithms/ppo/run_ppo.py`, which will reproduce PPO results on Procgen.


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