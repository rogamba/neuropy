# NeuroPy
 
 NeuroPy is a python library that uses a simplified implementation of the NEAT (Neuro Evolution of Augmented Topologies) algorithm published by Kenneth O. Stanley in 2002, which basically proposes a genetic algorithm for evolving neural networks. The main considerations of this algorithm vs other reinforcement learning algorithms are:
* Starts the evolution process with a population of individuals that have minimal structures/topologies
* Protects topological innovation (new network arrangements) through speciation
* Uses historical markings as a way around the competing conventions problem (two or more encoded networks can represent the same solution)  

This repo is intended to give an introduction and better understanding of NEAT Algorithm, inspired in the [neat-python](https://github.com/CodeReclaimers/neat-python) library by @CodeReclaimers with various modifications for the sake of simplicity.


## Prototypes

Neat can be trained to control or solve different models. The basic structure where the simulations reside is inside the /prototypes directory. 

Every simulation is related to a neat configuration file that must be inside the custom prototype directory.

If the new prototype requires a one or more models and classes you can include them also inside your new prototype directory in order to use them in the __main__.py file of the new prototype which will run the evolution process.


### Adding a simulation

You can add and test your own simulation by adding a new directory inside the /simulations folder. If your simulation requires a custom model to be tested, you must add your model inside the /models folder. The basic structure of the new files should be:
```shell
/prototype
  /new_prototype
    __main__.py
    model.py
    config.json
    /results
      winner.json
```

#### Files explanation:
* **config.json**: NEAT configuration parametes for the evolution process, this must be tunned up according to the complexity of the model and simulation.
* **__main.py__**: This file runs the main evolution process, must contain the function of genome evaluation. 
* **model.py**: Main file of the model to be tested, normally here would be the methods of the behaviours of the object we want to analyze, for example Finiancial models, Physical models, etc...
* **/results**: Directory that will store the results of the winner genome topology as a graph and as a json file


### Running the simulations

#### XOR
From the app root directory, execute:
```shell
$ python -m prototypes.xor
```

#### Single Pendulum Cart
From the app root directory, execute:
```shell
$ python -m prototypes.pendulum
```

#### Double Pendulum Cart
From the app root directory, execute:
```shell
$ python -m prototypes.double_pendulum
```


## Testing results

To test the winner genome and redo a simulation you can include a test_solution method in the main file of your prototype. You can run it with:

```shell
$> python -m prototypes.double_pendulum solution
```


# To Do
- Improve general performace of the package
- Improve neural_net module, implementing numpy and matrix multiplication instead of loops

