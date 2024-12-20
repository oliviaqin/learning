from typing import Callable, List, Any, Optional
from numpy.typing import NDArray

import random
import numpy as np

class DifferentialEvolution:
    """
    Class that performs differential evolution algorithm

    """

    def __init__(self,
        bounds: NDArray[np.float64], 
        pop_size: int,
        F: float, 
        cr: float, 
        obj: Optional[Callable] = lambda x: x):

        self.pop_size = pop_size
        self.bounds = bounds
        self.F = F
        self.cr = cr
        self.obj = obj

        self._validate_parameters()
    

    def _validate_parameters(self):
        assert pop_size >= 0, ValueError(
            f"Population size must not be negative. {pop_size} is not a valid population size."
        )
        assert F <= 2 and F >= 0, ValueError(
            f"{F} is not a valid parameter. Must be within the range [0,2]."
        )
        assert cr <= 1 and cr >= 0, ValueError(
            f"{cr} is not a valid cross over rate. Must be within the range [0,1]."
        )
    

    def select_random_vectors(self, population: NDArray[np.float64], j: int) -> NDArray[np.float64]:
        valid_members = [population[i] for i in range(self.pop_size) if i != j]
        indices = np.random.choice(range(len(valid_members)), size=3, replace=False)
        return np.array([valid_members[i] for i in indices])

    
    def mutate(self, x):
        return x[0] + self.F * (x[2] - x[1])

    
    def recombinate(self, individual: NDArray[np.float64], mutated_vector: NDArray[np.float64]):
        p = np.random.rand(len(individual))
        return np.where(p < self.cr, individual, mutated_vector)
    

    def run(self, num_iter: int = 100):
        """
        Runs the differential evolution algorithm for num_iter generations
        """
        population = bounds[:, 0] + np.random.rand(pop_size, len(bounds))*(bounds[:, 1] - bounds[:, 0])

        pop_best_all = []

        for i in range(num_iter):

            print(f"----------- Starting generation {i} -----------")

            gen_obj = []

            print(f"Iterating through individuals in population.....")

            for j in range(pop_size):

                # The individual in the population
                individual = population[j]

                # Select random vectors for a given individual
                random_vectors = self.select_random_vectors(population, j)

                # Mutate 
                mutated_vector = self.mutate(random_vectors)

                # Clip
                mutated_vector = [np.clip(mutated_vector[i], a_min=bounds[i,0], a_max=bounds[i,1]) for i in range(len(mutated_vector))]

                # Recombination
                recombined_vector = self.recombinate(individual, mutated_vector)

                # Calculate obj function
                recombined_vector_obj = self.obj(recombined_vector)
                individual_obj = self.obj(individual)

                # Greedy selection
                if recombined_vector_obj < individual_obj: 
                    population[j] = recombined_vector
                    gen_obj.append(recombined_vector_obj)
                else: 
                    gen_obj.append(individual_obj)
            
            gen_best = min(gen_obj)
            pop_best = population[gen_obj.index(min(gen_obj))]

            pop_best_all.append(pop_best) 


            print(f"----------- Finished generation {i} -----------")
            print(f"Best objective function value: {gen_best}")
            print(f"Best individual in population: {pop_best}")
        
        return pop_best_all


def ackley(x):
    """
    Ackley function.
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)

    term1 = -a * np.exp(-b * np.sqrt((1/n) * np.sum(x**2)))
    term2 = -np.exp((1/n) * np.sum(np.cos(c * x)))
    return term1 + term2 + a + np.exp(1)


# Define the Ackley function
def ackley_function(x, y):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + np.e + 20


def plot_ackley_optimization(best_vectors):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    x, y = np.meshgrid(x, y)

    # Evaluate the function
    z = ackley_function(x, y)

    # Create a 3D plot
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(x, y, z, cmap='coolwarm', edgecolor='none', antialiased=True, rstride=1, cstride=1, lw=0.5)

    # Extracting x, y coordinates from the best_vectors and computing their z-values
    best_x = best_vectors[:, 0]
    best_y = best_vectors[:, 1]
    best_z = ackley_function(best_x, best_y)

    # Plot the best vectors on the surface plot
    ax.scatter(best_x, best_y, best_z, color='black', marker='o', s=50, label='Best Vectors')

    # Add contour lines for better perception of depth
    ax.contour(x, y, z, zdir='z', offset=0, cmap=cm.coolwarm, linestyles="solid")

    # Add a color bar for reference
    colorbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    colorbar.set_label('Function Value')

    # Use a better view angle
    ax.view_init(elev=45, azim=235)

    # Set labels with larger fonts
    ax.set_title('Ackley Function', fontsize=16, pad=20)
    ax.set_xlabel('X-axis', fontsize=12, labelpad=10)
    ax.set_ylabel('Y-axis', fontsize=12, labelpad=10)
    ax.set_zlabel('Z-axis', fontsize=12, labelpad=10)

    # Show the plot
    plt.show()


if __name__ == "__main__":

    pop_size = 100
    bounds = np.array([(-5,5),(-5,5)])
    obj_func = ackley

    F = 0.5
    cr = 0.7

    diff_ev = DifferentialEvolution(
    pop_size=pop_size, 
    bounds=bounds,
    obj=obj_func,
    F=F, 
    cr=cr
    )


    best_vectors = diff_ev.run(num_iter=100)

    plot_ackley_optimization(np.array(best_vectors))

