import numpy as np

class ParticleSwarmOptimizer:
    
    def __init__(self, num_particles, dimensions, max_iter, probabilities, fitness_fn, w=0.5, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.probabilities = probabilities
        self.fitness_fn = fitness_fn
        self.w, self.c1, self.c2 = w, c1, c2

        self.positions = np.random.uniform(1, 254, (num_particles, dimensions))
        self.velocities = np.random.uniform(-5, 5, (num_particles, dimensions))
        
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.array([self.fitness_fn(p, probabilities) for p in self.positions])
        
        best_idx = np.argmax(self.pbest_scores)
        self.gbest_position = np.copy(self.pbest_positions[best_idx])
        self.gbest_score = self.pbest_scores[best_idx]
        self.history = []

    def optimize(self):
        for _ in range(self.max_iter):
            r1 = np.random.rand(self.num_particles, self.dimensions)
            r2 = np.random.rand(self.num_particles, self.dimensions)

            cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
            social = self.c2 * r2 * (self.gbest_position - self.positions)
            self.velocities = self.w * self.velocities + cognitive + social
            self.positions = np.clip(self.positions + self.velocities, 1, 254)

            for i in range(self.num_particles):
                score = self.fitness_fn(self.positions[i], self.probabilities)
                if score > self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = np.copy(self.positions[i])
                    if score > self.gbest_score:
                        self.gbest_score = score
                        self.gbest_position = np.copy(self.positions[i])

            self.history.append(self.gbest_score)

        return np.sort(np.round(self.gbest_position).astype(int)), self.history


class GeneticAlgorithm:
    
    def __init__(self, pop_size, dimensions, max_iter, probabilities, fitness_fn, mutation_rate=0.1):
        self.pop_size = pop_size
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.probabilities = probabilities
        self.fitness_fn = fitness_fn
        self.mutation_rate = mutation_rate

        self.population = np.random.uniform(1, 254, (pop_size, dimensions))
        self.history = []

    def optimize(self):
        for _ in range(self.max_iter):
            scores = np.array([self.fitness_fn(ind, self.probabilities) for ind in self.population])
            best_idx = np.argmax(scores)
            self.history.append(scores[best_idx])

            selected = []
            for _ in range(self.pop_size):
                i, j = np.random.choice(self.pop_size, 2, replace=False)
                selected.append(self.population[i] if scores[i] > scores[j] else self.population[j])
            selected = np.array(selected)

            next_pop = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = selected[i], selected[(i+1) % self.pop_size]
                cp = np.random.randint(1, self.dimensions) if self.dimensions > 1 else 0
                c1 = np.concatenate([p1[:cp], p2[cp:]])
                c2 = np.concatenate([p2[:cp], p1[cp:]])
                next_pop.extend([c1, c2])
            
            self.population = np.array(next_pop)[:self.pop_size]
            mutation_mask = np.random.rand(*self.population.shape) < self.mutation_rate
            noise = np.random.normal(0, 5, self.population.shape)
            self.population = np.clip(self.population + mutation_mask * noise, 1, 254)

        final_scores = np.array([self.fitness_fn(ind, self.probabilities) for ind in self.population])
        best_ind = self.population[np.argmax(final_scores)]
        return np.sort(np.round(best_ind).astype(int)), self.history