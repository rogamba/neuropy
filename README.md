# NeuroPy
 
 NeuroPy is a python library that uses a simplified implementation of the NEAT (Neuro Evolution of Augmented Topologies) algorithm published by Kenneth O. Stanley in 2002, which basically proposes a genetic algorithm for evolving neural networks. The main considerations of this algorithm vs other reinforcement learning algorithms are:
* Starts the evolution process with a population of individuals that have minimal structures/topologies
* Protects topological innovation (new network arrangements) through speciation
* Uses historical markings as a way around the competing conventions problem (two or more encoded networks can represent the same solution)  

This repo is intended to give an introduction and better understanding of NEAT Algorithm, inspired in the [neat-python](https://github.com/CodeReclaimers/neat-python) library by @CodeReclaimers with various modifications for the sake of simplicity.

## Simulations

Neat can be trained to control or solve different models. The basic structure where the simulations reside is inside the /simulations directory. Every simulation is related to a neat configuration file that must be in the custom simulation directory. If the simulation needs to run and test a model, you have to import it from the /models directory.

### Adding a simulation

You can add and test your own simulation by adding a new directory inside the /simulations folder. If your simulation requires a custom model to be tested, you must add your model inside the /models folder. The basic structure of the new files should be:
```shell
/simulations
  /my_simulation
    config.json
    /plots
    __main__.py
/models
    mymodel.py
```

#### Files explanation:
* config.json:  all the neat configuration parametes, this must be tunned up according to the complexity of the model and simulation
* main.py: this file runs the main process, must contain the function of genome evaluation 
* /plots: where the winning net topology will be dumped
* mymodel.py: module that contains classes and/or functions needed to test your own model


### Running the simulations

#### XOR
From the app root directory, execute:
```shell
$ python -m simulations.xor
```

#### Single Pendulum Cart
From the app root directory, execute:
```shell
$ python -m simulations.pendulum
```

#### Double Pendulum Cart
From the app root directory, execute:
```shell
$ python -m simulations.double_pendulum
```


# Modules and Classes


## Gene

## Phenotype

## Genome
Clase base que individualiza cada una de las soluciones del proceso evolutivo
#### Propiedades
* 
#### Métodos


## Population
Clase con las propiedades y métodos de la generación en curso, tiene métodos específicos para la reproducción y especiación de la generación actual.
#### Propiedades
```python
* config:             <dict> parámetros de configuración
* genomes:            <list> lista de objetos Genome, individuos de la generación actual
* champion:           <obj:Genome> objeto de Genome con el individuo con mejor desempeño de la generación
* ancestors:          <dict> generaciones pasadas
* generation:         <int> número de generación actual
* elitism             <int>
* size:               <int> tamaño de las poblaciones
```
#### Métodos
* init
  - Crea una nueva generación con individuos aleatorios
* new: Crea una nueva población con individuos aleatorios, con base en los parámetros de configuración iniciales
  - Agrega el número pop_size de individuos aleatorios, instancía un objeto de Genome por cada individuo
  - [speciate] Separa los individuos en distintas especies
* speciate: Itera los individuos de una población y los separa en especies por similaridad de especies
* set_genome_fitness: Itera los individuos de una población y aplica la fitness function para evaluar su desempeño. Establece el campeón de la generación
* reproduce: Itera los individuos para hacer el crossover de los genomas 

## Evolution
Clase que corre todo el proceso de evolución, en ella se definien
#### Propiedades
- config:             <dict> parámetros de configuración universales
- population:         <obj:Population> población actual del proceso evolutivo
- fitness_critereon:  <function> criterio de desempeño, puede ser (min, max, avg)
- champ:              <obj:Genome> genoma (solución) con mejor desempeño de todo el proceso evolutivo.
#### Métodos
* init: 
  - Crea la población de la generación 0
* run: corre todo el proceso de evolución
  - [population.set_genome_fitness] Evalúa la generación actual pasando la fitness function
  - verifica y establece al campeón global
  - Revisa el umbral de fitness para terminar la iteración si fue alcanzado
  - [population.reproduce]: establece en la variable de population la nueva generación,
  - [population.speciate]: Separa y agrupa los individuos de la generación en especies
  - IN: fitness_function, n
    - fitness_function: función de desempeño definida en el módulo donde se correrá la clase
    - n: número de generaciones
- OUT: champ
- EXP: Corre el proceso de evolución, el cuál se detendrá cuando el campeón global rebase el imbral de desempeño establecido en el archivo de configuración o se haya alcanzado el número máximo de generaciones a evolucionar





# Modelo de entidades


Evolution:
    properties
        .config             <dict>
        .population         <obj:NodeGene>
        .fitness_citerion   <function>
        .fitness_threshold  <float>
        .champion           <obj:Genome>
        .start              <datetime>
    methods
        .init
        .run
        .print_solution


Population:
    properties
        .config             <dict>
        .genomes            [<obj:Genome>]
        .ancestors          <dict> Save the parents of the gene with the same key { key: (parent1, parent2) }
        .species            <dict> { species : <genomes> }
        .species_fitness_func   <func>
        .genome_to_species  
        .champion           <genome>
        .generation         <int>
        .elitism            <int> How many of the best genomes are copied to the nex gen
        .pop_size           <int>
        .species_stagnation <int>
        .max_stagnation
        .species_elitism
        .min_species_size
        .survial_threshold
        .genome_indexer
        .species_indexer
    methods
        .init
        .new
        .evaluate 
        .set_genome_fitness
        .stagnation
        .speciate
        .reproduce
        .seeds              


Species:
    properties
        .config
        .key
        .created
        .last_improved
        .generation
        .representative
        .genomes
        .fitness_history
        .fitness
        .adjusted_fitness
    methods
        .update
        .get_fitness


Genome:
    properties
        .key
        .nodes              [<obj:NodeGene>]
        .edges              [<obj:EdgeGene>]
        .input_keys         <list>
        .output_keys        <list>
        .fitness            <float>
    methods
        .init
        .new                Configure new genome form the given configurations 
        .connect            Fully connect the nodes in the layers  
        .distance
        .size
        .mutate             Eval rand vs the probablilities for: add, delete node or connection, then mutate nodes and connections?
        .mutate_add_node
        .mutate_add_connection
        .mutate_delete_node
        .mutate_delete_connection
        .add_connection
        .add_node



NodeGene:
    properties:
        .key                <int>
        .bias               <float>
        .reponse            <float>
        .activation
        .aggregation
    methods:
        .distance
        .crossover          Creates a new gene, randomly selecting attributes form their parents   
        .mutate


EdgeGene:
    properties:
        .key                <tuple>
        .weight             <float>
        .enabled            <bool>
        .reponse            <float>
        .activation         <string>
        .aggregation        <string>
    methods:
        .distance
        .crossover          Creates a new gene, randomly selecting attributes form their parents   
        .mutate


Phenotype:
    properties:
        .inputs
        .outputs
        .edges
        .layers
        .weights
        .bias
        .activation
    methods:
        .get_path           Obtiene el path de nodos y conexiones necesarios desde el output
        .format
        .create             Returns the NeuralNetwork from the genotype


NeuralNet:
    properties:
      .layers
      .W
      .b
      .f
    methods:
      .activate


Indexer

DistanceCache

DoublePendulum


## Indexer
Entidades que necesitan un objeto de indexer para trackear keys:
Population.genome_indexer
Population.species_indexer


# Proceso de desarrollo de la clase
- Phenotype: Conversión del genoma al fenotipo (Red Neuronal)
  - Terminar la representación de la red en el genoma [no podrá ser de forma matricial]
- Terminar el setup de la población inicial
- Reproducción
- Mutación
- Especiación


# Pasos generales del proces
- Establecer parámetros de configuración
- Definir fitness function para evaluar a los genomas 
- Cargar los parámetros de configuración en variable global del paquete 
- Instanciar objeto de evolución, dentro del instanciamento de la evolución iniciamos objeto de la población
- Creamos cada uno de los genomas
- Especiamos población
- Definimos ancestros vacíos
- Corremos proceso de evolución
- Evaluamos cada uno de los genomas para obtener su fitness
- Obtenemos el phenotype de cada genoma y corremos feedforward para obtener respuesta del sistema
- Simulamos la fitness function y obtenemos fitness del genoma
- Evaluamos campeón global
- Reproducimos
- Especiamos



# Pasos generales
- Establecer parámetros en archivo de configuración
- Definir la función de desempeño con la que se evaluarán los genomas
- Cargar los parámetros de configuración en una variable global para todo el paquete
- Instanciamos objeto de evolución [run]:
  - Set fitness_criterion [Evolution] 
  - Set population (Instanciamos objeto de población) [Evolution]:
    - genomes=[], species, ancesters=[], elitism, survival_threshold [Poopulation]:  (establecer ancestro vacío por cada genome)
    - set genomes [Population]:
      - links={}, nodes={}, fitness=None, [Genome]
      - configure new genome based on the config params [Genome]: 
        - set init node
        - set init connections
- Corremos la evolución [run]
  - Evaluamos el desempeño de cada genome [Evolution > run]
    - Por cada genoma de la población, contruimos un fenotipo [Population > ] 
    - Evaluamos el fenotipo y obtenemos el fitness [Phenotype > evaluate]


    
- Establecer generación a 0, el mejor genome a None y especiar la población [Population, Genome]












- Empezar la evolución:
  - Evaluar cada uno de los individuos de la población:
    - Creación del phenotype a partir del genotype (nn)
    - Hacer feedforward de la red
    - Obtener su fitness con base en su desempeño
  





## Espaciación

- Se toma un representante de cada especie de manera aleatoria de la población anterior
- Se compara cada gen g de la población actual con los representantes de la generación pasada
- La comparación se realiza midiendo la distancia de compatibilidad entre g y el representante
- Se coloca al gen g en la primera generación con la que es compatible
- Si g no es compatible con ningún representante se crea una nueva especie
- Los individuos que pertenecen a especies muy grandes (con muchos individuos) con penalizados por la función



# To Do
- Improve general performace of the package
- Improve neural_net module, implementing numpy and matrix multiplication instead of loops

