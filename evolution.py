from population import Population
from pprint import pprint
from utils import *
import datetime


class Evolution(object):

    def __init__(self,config):
        ''' Inicia población, criterio de evaluación de desempeño
        '''
        print("[evolution] Init evolution")
        # Configuration
        self.config = config
        # Reproduction type
        # Population - set initial population
        self.population = Population(self.config)
        self.start = datetime.datetime.utcnow()
        #print(self.population)
        # Fitness criterion (min | max | avg) for the threshold evaluation
        self.fitness_critereon = eval(self.config['fitness_criterion'])
        self.fitness_threshold = self.config['fitness_threshold']
        # Best genome to none
        self.winner = None
    

    def run(self, fitness_function, generations=None):
        ''' Evolves the population n number of times, evaluating
            it with the given fitness function
        '''

        k=0

        self.population.print_species(self.config, self.population.genomes, self.population.species, k)
        
        # Iterate through the number of generations
        #print("[evolution] Evolving populations... %s, %s" % (k,generations))
        while generations is None or k < generations:
            k+=1

            self.population.generation = k

            #print("[evolution] Evolving generation %s - %s" % (k,generations))
            #print("[evolution] Setting fitness to population")
            # Set the fitness of the population -> Evaluate every genome in the current generation
            self.population.set_genome_fitness(fitness_function)


            # Track the winner of the entire evolution
            #print("[evolution] Updating winner")
            if self.winner == None or self.population.champion.fitness > self.winner.fitness:
                self.winner = self.population.champion
                #print("[evolution] >>>>>>>>  Setting winner with fitness "+str(self.winner.fitness))


            self.report()
                

            # End if the threshold fitness is reached fitness_critereon (min | max | avg) 
            #print("[evolution] Checking if solution found")
            fitness = self.fitness_critereon([ g.fitness for key,g in self.population.genomes.items() ])
            if fitness >= self.fitness_threshold:
                self.solution_found()
                break

            # Create the next generation from the current generation 
            # Pass he species from the previous generation (representatives)
            #print("[evolution] Reproduction")
            #print("[evolution] Previous generation fitness avg: " + str(sum([ g.fitness for g in self.population.genomes.values() ])/len([ g for g in self.population.genomes ])) )
            self.population.reproduce()
            #print("[evolution] New generation: ")
            #pprint(self.population)

            # Divide the new population into species
            #print("[evolution] Speciation, gen: %s" % (k))
            self.population.speciate()
            # Print the species and attributes
            #for i,s in self.population.species.items():
            #    print(s)

            #self.population.print_species(self.config, self.population.genomes, self.population.species, k)



        self.print_solution()

        return self.winner


    def solution_found(self):
        ''' Actions to be made when solution is found
        '''
        self.time = datetime.datetime.utcnow()-self.start
        print("[evolution] >>>>>>>> Solution found at generation %s, process took %s " % (self.population.generation,str(self.time)))
        return



    def report(self):
        ''' Print statistics about the current generation
        '''
        fitnesses = [ g.fitness for g in self.population.genomes.values() ]
        print("[evolution] Mean generation fitness: "+str(mean(fitnesses)))
        print("Species: "+ str(len(self.population.species)))
        print("Best genome fitness: "+ str(self.winner.fitness))
        pass



    def print_solution(self):
        print("[evolution] Best genome of the evolution process")
        print(self.winner)
        return