# thesis-mlcg

---

## What is in here

Configuration and tooling: yaml files.

## Last update:
```
datasets/
  bba/
    pipeline/      the six pipeline configs + the step-1b watcher
    sim/
      prioronly/   Langevin runs with the prior alone
      trained/     Langevin runs with the trained SchNet + prior
  trpcage_demo/    the toy set used to learn the pipeline

tools/
  priors/   fit diagnostics          check_priors.py # <-fit plots , prior_fit_check.py,
                                     coupling_test.py, worked_example.py
  tica/     free energy surfaces     project_sim.py, plot_fes.py        + README
  viz/      just to bring trajectories and visualise them, +readme
  theme/    the plotting theme       thesis_style.py                    + README
  tmux/     session status line

logs/       stdout from each pipeline step: for now just bba folder
training/ 
  bba/  SchNet training config and launcher
```
