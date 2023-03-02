# `ilastik-feature-selection` example notebooks

## Where to start?

### Create an environment with all needed dependencies

You can use `conda`/`mamba` with the `environment.yml` file in this folder in order to run the example:
We recommend `mamba` as it creates environments faster.

```bash
# from within the `example` folder:

# 1. create the environment
mamba env create -n feature-selection --file environment.yml

# 2. activate the envrionment
mamba activate feature-selection

# 3. start the jupyter notebook server
# which should open your default browser
jupyter notebook --notebook-dir=.
```

### Check out the notebook

The main use-case is documented in the `feature-selection-example.ipynb` notebook.
There you'll learn how to use the feature selection methods provided by this package to reduce the feature set for a `sklearn` Random Forest.

The `feature-selection-example-using-vigra.ipynb` notebook shows how to use these methods with a `vigra` Random Forest, for those who want to know :).
