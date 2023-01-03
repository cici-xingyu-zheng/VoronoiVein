### Clone the repository to local:
`clone https://github.com/cici-xingyu-zheng/VoronoiVein.git`


### Create virtual environment for this project:

#### 1. using Conda (recommended):

1. navigate to the project folder, create a virtual environment (called `pilea`):

`conda env create -f environment.yml`
or
`conda create --name pilea --file pilea_requirements.txt`

This way we create a virtual environment with all the package versions freezed for this project.

2. activate the `pilea` environment we just created:

`conda activate pilea`

#### 2. alternatively, using pip:
`pip install -r pilea_requirements.txt`

This way we install all the dependencies this project needs. But some packages might be of different version than your local ones.

### Run the demo code

One way to try out the code is to open the demo notebooks on a web-browser:
`jupyter notebook`
