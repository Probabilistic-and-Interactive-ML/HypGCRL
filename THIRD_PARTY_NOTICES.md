# Third-party notices

This repository contains code adapted from the projects below. Each keeps its original license.
Every adapted source file starts with a header that names its source and states that it was modified for this repository.

| Component | Source | Copyright | License |
|---|---|---|---|
| Training framework and environments (`jaxgcrl/envs/`, replay buffer, evaluator, online training loop) | [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) | Copyright 2023 FLAIR | Apache-2.0 (text in `LICENSE`) |
| Environment base classes in `jaxgcrl/envs/` | [Brax](https://github.com/google/brax) | Copyright The Brax Authors | Apache-2.0 (text in `LICENSE`) |
| Maze construction in the maze environments | [D4RL](https://github.com/Farama-Foundation/D4RL) | Copyright 2022 Farama Foundation | Apache-2.0 (text in `LICENSE`) |
| Residual networks, maze layouts, `ant_maze_generalization.py` | [scaling-crl](https://github.com/wang-kevin3290/scaling-crl) | Copyright 2023 FLAIR | Apache-2.0 (text in `LICENSE`) |
| `jaxgcrl/offline/`: datasets, evaluation, environment helpers, networks, learner, training loop | [OGBench](https://github.com/seohongpark/ogbench) | Copyright (c) 2024 OGBench Authors | MIT (below) |
| Franka Emika Panda model (`jaxgcrl/envs/assets/franka_emika_panda/`) | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | see the `LICENSE` in that folder | Apache-2.0 (see the `LICENSE` in that folder) |

## OGBench

```
The MIT License (MIT)

Copyright (c) 2024 OGBench Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
