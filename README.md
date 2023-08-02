Below are the steps to explore the Voronoi tests with Pilea leaf samples.


### Clone the repository to local:

```
git clone https://github.com/cici-xingyu-zheng/VoronoiVein.git
```


### Create virtual environment for this project:

#### 1. using Conda (recommended):

1. navigate to the project folder, create a virtual environment (called `pilea`):

```
conda env create -f pilea_env.yml
```

or

```
conda create --name pilea --file pilea_requirements.txt
```

This way we create a virtual environment with all the package versions freezed for this project.


2. activate the `pilea` environment we just created:

```
conda activate pilea
conda install jupyter 
```


#### 2. alternatively, using `pip`:

```
pip install -r pilea_requirements.txt
```

This way we install all the dependencies this project needs. But some packages might be of different versions than your local ones.

### Run the demo code

The easiest way to try out the code locally is to open the demo notebooks on a web-browser. On command line, run

```
jupyter notebook
```

and navigate to notebooks like `voronoi_tests/notebooks/voronoi_tests_example.ipynb` and run them.

