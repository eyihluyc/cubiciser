from fragments import select, mutate, fitness_function, crossover, pair_as_parents, decorated, initialize_population
from imagefarm import load, save
from functools import partial

ITERATIONS = 90
POPULATION_SIZE = 9
MUTATION_PROBABILITY = 0.4
OFFSPRINGS = POPULATION_SIZE // 3
SELECTION_PROBABILITY = 0.85
PARENTSHIP_PROBABILITY = 0.3

images = load()
population = initialize_population(images, POPULATION_SIZE)

fitness_dict = {}

for i in range(ITERATIONS):
    for j in range(POPULATION_SIZE):
        # imshow(f"{i}-{j}", decode(population[j]))
        print(fitness_function(population[j], fitness_dict))
    print()
    # waitKey()
    # destroyAllWindows()
    mutants = map(partial(mutate, p=MUTATION_PROBABILITY), population)
    crossovers = map(crossover, pair_as_parents(population, OFFSPRINGS, PARENTSHIP_PROBABILITY, fitness_dict))
    population = select(SELECTION_PROBABILITY, [*mutants, *crossovers], POPULATION_SIZE, fitness_dict)


for j in range(POPULATION_SIZE):
    # imshow(f"{i}-{j}", decode(population[j]))
    save(decorated(population[j]), identifier=333)


save(decorated(select(1, population, 1, fitness_dict)[0]), identifier=13)
