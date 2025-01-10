# DRLjellyfish
Code for DRL control of a 2D jellyfish
arXiv: https://arxiv.org/abs/2409.08815
Platform: Ubuntu2204 on WSL2, Win11
Software: IBAMR v0.11 installed on wsl2 for CFD, need to modify source code and compile: change StandardIBForceGen.cpp to add tangent force on muscle and change beam force from non-inv beam to torsional spring
          pytorch for training
Training: python 3.10, DQN training needs multiple times to train well, SAC one seems to have less effort
