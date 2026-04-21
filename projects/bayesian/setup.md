# Environment Setup

These are notes for how to setup a dedicated environment for the code example. This needs to be done
just once. If you have an old environment that is somehow broken, you can remove it with the
instructions the "Conda Environment Removal" section.

 Install Miniconda if it isn't already installed.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run the setup and accept
bash Miniconda3-latest-Linux-x86_64.sh
```

Create a new conda environment
```
conda env create -f environment.yml
```

# Running the Example Code

A fast one-liner to run the code example is the following.

```bash
conda run -n bayesian_example python plate_modeling.py
```

Alternatively, you can activate the conda environment.

```bash
conda activate bayesian_example
```

Run the example 
```python
 python plate_modeling.py
```

Finally, deactivate when done.

```bash
conda deactivate
```

# Removing

The environment can fully be removed with the following. 

```bash
conda remove --name bayesian_example --all
```