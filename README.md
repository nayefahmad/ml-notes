# ML notes

## Contents 
1. [PyTorch demo: image classification](https://github.com/nayefahmad/ml-notes/blob/main/src/pytorch-demo.ipynb)
2. [Experimenting with simple random sampling versus stratified sampling](https://github.com/nayefahmad/ml-notes/blob/main/src/simple-random-and-stratified-sampling.ipynb)

## Repo structure 

- `src` directory: code files 
- `.pre-commit-config.yaml`: config for use with `pre-commit`. It specifies what hooks to use. 
  Once this file is created, if you run `pre-commit install`, the pre-commit tool will populate the 
  `pre-commit` file in the `./.git/hooks` directory. Helpful references: 
    - [Automate Python workflow using pre-commits: black and flake8](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/)
    - [Keep your code clean using Black & Pylint & Git Hooks & Pre-commit](https://towardsdatascience.com/keep-your-code-clean-using-black-pylint-git-hooks-pre-commit-baf6991f7376)
    - [pre-commit docs](https://pre-commit.com/#)
- `.flake8`: config for Flake8. Mainly used to specify max-line-length=88, to match [Black's default](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
- `requirements.txt`: python packages used 


