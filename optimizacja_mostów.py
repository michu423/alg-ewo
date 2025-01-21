#Rojek, Kuczaj

import random
import matplotlib.pyplot as plt
import numpy as np

# zakresy dla zmiennych
MIN_LENGTH, MAX_LENGTH = 10, 100
MIN_HEIGHT, MAX_HEIGHT = 1, 20
MIN_SUPPORT_DIST, MAX_SUPPORT_DIST = 5, 50
MIN_ANGLE, MAX_ANGLE = 5, 30

# parametry
experiments = [
    {"POPULATION_SIZE": 50, "CROSSOVER_PROBABILITY": 0.8, "MUTATION_PROBABILITY": 0.01, "NUM_GENERATIONS": 500},
    {"POPULATION_SIZE": 100, "CROSSOVER_PROBABILITY": 0.9, "MUTATION_PROBABILITY": 0.02, "NUM_GENERATIONS": 500},
    {"POPULATION_SIZE": 30, "CROSSOVER_PROBABILITY": 0.7, "MUTATION_PROBABILITY": 0.05, "NUM_GENERATIONS": 300}
]

# funkcje alg
def initialize_population(size):
    return [[random.uniform(MIN_LENGTH, MAX_LENGTH),
             random.uniform(MIN_HEIGHT, MAX_HEIGHT),
             random.uniform(MIN_SUPPORT_DIST, MAX_SUPPORT_DIST),
             random.uniform(MIN_ANGLE, MAX_ANGLE)] for _ in range(size)]

def evaluate(individual):
    weight = sum(individual)
    strength = 1000 - sum(i ** 2 for i in individual)
    return weight if strength > 0 else float('inf')

def tournament_selection(population, size):
    return [min(random.sample(population, size), key=evaluate) for _ in range(len(population))]

def crossover(parent1, parent2, prob):
    if random.random() > prob:
        return parent1[:], parent2[:]
    point = random.randint(1, len(parent1) - 1)
    return (parent1[:point] + parent2[point:], parent2[:point] + parent1[point:])

def mutate(individual, prob):
    if random.random() < prob:
        idx = random.randint(0, len(individual) - 1)
        individual[idx] = random.uniform(
            [MIN_LENGTH, MIN_HEIGHT, MIN_SUPPORT_DIST, MIN_ANGLE][idx],
            [MAX_LENGTH, MAX_HEIGHT, MAX_SUPPORT_DIST, MAX_ANGLE][idx]
        )
    return individual

def genetic_algorithm(params):
    population = initialize_population(params["POPULATION_SIZE"])
    stats = {'Najlepszy': [], 'Najgorszy': [], 'Średni': []}

    for _ in range(params["NUM_GENERATIONS"]):
        fitness_values = [evaluate(ind) for ind in population]
        stats['Najlepszy'].append(min(fitness_values))
        stats['Najgorszy'].append(max(fitness_values))
        stats['Średni'].append(sum(fitness_values) / len(fitness_values))

        selected = tournament_selection(population, 3)
        next_population = []
        for i in range(0, len(selected), 2):
            offspring1, offspring2 = crossover(selected[i], selected[(i + 1) % len(selected)],
                                               params["CROSSOVER_PROBABILITY"])
            next_population.extend([mutate(offspring1, params["MUTATION_PROBABILITY"]),
                                    mutate(offspring2, params["MUTATION_PROBABILITY"])])
        population = next_population
    return stats

# wykresy
for idx, exp_params in enumerate(experiments):
    all_runs = {'Najlepszy': [], 'Najgorszy': [], 'Średni': []}
    for _ in range(10):  # 10 uruchomien
        stats = genetic_algorithm(exp_params)
        for key in all_runs:
            all_runs[key].append(stats[key])

    # obl srednich i dchylen dla kazdej gen
    generations = range(exp_params["NUM_GENERATIONS"])
    plt.figure(figsize=(10, 6))
    for key, color in zip(['Najlepszy', 'Najgorszy', 'Średni'], ['green', 'red', 'blue']):
        data = np.array(all_runs[key])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plt.plot(generations, mean, label=f'{key} fitness', color=color)
        plt.fill_between(generations, mean - std, mean + std, alpha=0.2, color=color)

    plt.xlabel('Generacja')
    plt.ylabel('Fitness')
    plt.title(f'Wyniki algorytmu ewolucyjnego (Eksperyment {idx + 1})')
    plt.legend()
    plt.show()