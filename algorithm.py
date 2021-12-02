import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import clone

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_crossover, get_mutation, get_sampling, get_termination, get_selection
from operator import itemgetter

import pandas as pd
import pickle
import os


def distance(x, y):
    return np.sum(np.abs(x - y))


def taxicab_sample(n, r):
    sample = []

    for _ in range(n):
        spread = r - np.sum([np.abs(x) for x in sample])
        sample.append(spread * (2 * np.random.rand() - 1))

    return np.random.permutation(sample)


class AECCR:
    def __init__(self, X, y, scaling=0.0, n=None):
        self._X = X
        self._y = y
        self.scaling = scaling
        self.n = n

        self.set_distances()

    def set_distances(self):
        self._classes = np.unique(self._y)
        sizes = [sum(self._y == c) for c in self._classes]

        assert len(self._classes) == len(set(sizes)) == 2

        self._minority_class = self._classes[np.argmin(sizes)]
        self._majority_class = self._classes[np.argmax(sizes)]
        self._minority = self._X[self._y == self._minority_class]
        self._majority = self._X[self._y == self._majority_class]

        if self.n is None:
            self._n = len(self._majority) - len(self._minority)

        self._distances = np.zeros((len(self._minority), len(self._majority)))

        for i in range(len(self._minority)):
            for j in range(len(self._majority)):
                self._distances[i][j] = distance(self._minority[i], self._majority[j])

    def fit_sample(self, energy=0.25):

        if not isinstance(energy, np.ndarray):
            energy = np.array([energy for i in range(self._minority.shape[0])])

        if np.sum(energy) == 0:
            return self._X, self._y

        energy = energy * (self._X.shape[1] ** self.scaling)

        radii = np.zeros(len(self._minority))
        translations = np.zeros(self._majority.shape)

        majority = np.copy(self._majority)
        minority = np.copy(self._minority)

        for i in range(len(minority)):
            minority_point = minority[i]
            remaining_energy = energy[i]
            r = 0.0
            sorted_distances = np.argsort(self._distances[i])
            current_majority = 0

            while True:
                if current_majority == len(majority):
                    if current_majority == 0:
                        radius_change = remaining_energy / (current_majority + 1.0)
                    else:
                        radius_change = remaining_energy / current_majority

                    r += radius_change

                    break

                radius_change = remaining_energy / (current_majority + 1.0)

                if self._distances[i, sorted_distances[current_majority]] >= r + radius_change:
                    r += radius_change

                    break
                else:
                    if current_majority == 0:
                        last_distance = 0.0
                    else:
                        last_distance = self._distances[i, sorted_distances[current_majority - 1]]

                    radius_change = self._distances[i, sorted_distances[current_majority]] - last_distance
                    r += radius_change
                    remaining_energy -= radius_change * (current_majority + 1.0)
                    current_majority += 1

            radii[i] = r

            for j in range(current_majority):
                majority_point = majority[sorted_distances[j]]
                d = self._distances[i, sorted_distances[j]]

                if d < 1e-20:
                    dif = (1e-6 * np.random.rand(len(majority_point)) + 1e-6) * \
                                      np.random.choice([-1.0, 1.0], len(majority_point))
                    majority_point += dif
                    d = distance(minority_point, majority_point)

                translation = (r - d) / d * (majority_point - minority_point)

                translations[sorted_distances[j]] += translation

        majority += translations

        appended = []

        for i in range(len(minority)):
            minority_point = minority[i]
            synthetic_samples = int(np.round(1.0 / (radii[i] * np.sum(1.0 / radii)) * self._n))
            r = radii[i]

            for _ in range(synthetic_samples):
                appended.append(minority_point + taxicab_sample(len(minority_point), r))

        return np.concatenate([majority, minority, appended]), \
               np.concatenate([np.tile([self._majority_class], len(majority)),
                               np.tile([self._minority_class], len(minority) + len(appended))])


class PymooProblem(ElementwiseProblem):
    def __init__(self, n_var, aeccr, classifier, X_train, y_train, X_test, y_test, measures):
        self.n_var = n_var
        self.measures = measures
        self.classifier = classifier
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.aeccr = aeccr
        super().__init__(n_var=self.n_var,
                         n_obj=len(measures),
                         n_constr=0,
                         xl=np.full((n_var,), 0.0),
                         xu=np.full((n_var,), 1.0))

    def _evaluate(self, x, out, *args, **kwargs):
        new_X, new_y = self.aeccr.fit_sample(x)
        c = clone(self.classifier)
        try:
            c.fit(new_X, new_y)
        except:
            print(x)
            df_energy = pd.DataFrame(x)
            df_X = pd.DataFrame(self.X_train)
            df_y = pd.DataFrame(self.y_train)
            df_energy.to_csv('blad_energy.csv')
            df_X.to_csv('blad_X.csv')
            df_y.to_csv('blad_y.csv')

        y_pred = c.predict(self.X_test)
        measures_values = [-m(self.y_test, y_pred) for m in self.measures]
        out["F"] = measures_values


class MOO_CCRSelection:
    def __init__(self, classifier, measures, n_splits=1, energies=(0.25,), scaling_factors=(0.0,), n=None, criteria=['best'], test_size=0.5, save_directory='exp'):
        self.classifier = classifier
        self.measures = measures
        self.n_splits = n_splits
        self.energies = energies
        self.scaling_factors = scaling_factors
        self.n = n
        self.criteria = criteria
        self.save_directory = save_directory
        self.selected_energy = None
        self.selected_scaling = None
        self.sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)

    def pick_solutions(self, results, criteria):
        def pick_best(solutions, objectives, index):
            return solutions[objectives.index(max(objectives, key=itemgetter(index)))]

        def pick_balanced(solutions, objectives):
            return solutions[objectives.index(min(objectives, key=lambda i: abs(i[0] - i[1])))]

        solutions = results.X
        objectives = [tuple(-obj) for obj in results.F]

        picked_solutions = []

        for criterion in criteria:
            if criterion == 'best':
                for i in range(len(objectives[0])):
                    picked_solutions.append(pick_best(solutions, objectives, i))
            elif criterion == 'balanced':
                picked_solutions.append(pick_balanced(solutions, objectives))
        return picked_solutions


    def fit_sample(self, X, y):
        self.sss.get_n_splits(X, y)
        classes = np.unique(y)

        X = X.astype('float32')
        y = y.astype('float32')

        for train_idx, test_idx in self.sss.split(X, y):
            sizes = [sum(y[train_idx] == c) for c in classes]
            aeccr = AECCR(X[train_idx], y[train_idx])
            problem = PymooProblem(min(sizes), aeccr, self.classifier, X[train_idx], y[train_idx], X[test_idx], y[test_idx], self.measures)
            algorithm = NSGA2(
                pop_size=100,
                sampling=get_sampling("real_random"),
                crossover=get_crossover("real_ux"),
                mutation=get_mutation("real_pm"),
                eliminate_duplicates=True
            )
            termination = get_termination("n_gen", 100)
            res = minimize(problem,
                           algorithm,
                           termination,
                           seed=1,
                           save_history=True,
                           verbose=True)

        filename = os.path.join(self.save_directory, "optimisation_results")
        outfile = open(filename, 'wb')
        pickle.dump(res, outfile)
        outfile.close()

        solutions = self.pick_solutions(res, self.criteria)

        return [aeccr.fit_sample(energy) for energy in solutions]


class CCR:
    def __init__(self, energy=0.25, scaling=0.0, n=None):
        self.energy = energy
        self.scaling = scaling
        self.n = n

    def fit_sample(self, X, y):
        classes = np.unique(y)
        sizes = [sum(y == c) for c in classes]

        assert len(classes) == len(set(sizes)) == 2

        minority_class = classes[np.argmin(sizes)]
        majority_class = classes[np.argmax(sizes)]
        minority = X[y == minority_class]
        majority = X[y == majority_class]

        if self.n is None:
            n = len(majority) - len(minority)
        else:
            n = self.n

        energy = self.energy * (X.shape[1] ** self.scaling)

        distances = np.zeros((len(minority), len(majority)))

        for i in range(len(minority)):
            for j in range(len(majority)):
                distances[i][j] = distance(minority[i], majority[j])

        radii = np.zeros(len(minority))
        translations = np.zeros(majority.shape)

        for i in range(len(minority)):
            minority_point = minority[i]
            remaining_energy = energy
            r = 0.0
            sorted_distances = np.argsort(distances[i])
            current_majority = 0

            while True:
                if current_majority == len(majority):
                    if current_majority == 0:
                        radius_change = remaining_energy / (current_majority + 1.0)
                    else:
                        radius_change = remaining_energy / current_majority

                    r += radius_change

                    break

                radius_change = remaining_energy / (current_majority + 1.0)

                if distances[i, sorted_distances[current_majority]] >= r + radius_change:
                    r += radius_change

                    break
                else:
                    if current_majority == 0:
                        last_distance = 0.0
                    else:
                        last_distance = distances[i, sorted_distances[current_majority - 1]]

                    radius_change = distances[i, sorted_distances[current_majority]] - last_distance
                    r += radius_change
                    remaining_energy -= radius_change * (current_majority + 1.0)
                    current_majority += 1

            radii[i] = r

            for j in range(current_majority):
                majority_point = majority[sorted_distances[j]]
                d = distances[i, sorted_distances[j]]

                if d < 1e-20:
                    majority_point += (1e-6 * np.random.rand(len(majority_point)) + 1e-6) * \
                                      np.random.choice([-1.0, 1.0], len(majority_point))
                    d = distance(minority_point, majority_point)

                translation = (r - d) / d * (majority_point - minority_point)
                translations[sorted_distances[j]] += translation

        majority += translations

        appended = []

        for i in range(len(minority)):
            minority_point = minority[i]
            synthetic_samples = int(np.round(1.0 / (radii[i] * np.sum(1.0 / radii)) * n))
            r = radii[i]

            for _ in range(synthetic_samples):
                appended.append(minority_point + taxicab_sample(len(minority_point), r))

        return np.concatenate([majority, minority, appended]), \
               np.concatenate([np.tile([majority_class], len(majority)),
                               np.tile([minority_class], len(minority) + len(appended))])


class CCRSelection:
    def __init__(self, classifier, measure, n_splits=5, energies=(0.25,), scaling_factors=(0.0,), n=None):
        self.classifier = classifier
        self.measure = measure
        self.n_splits = n_splits
        self.energies = energies
        self.scaling_factors = scaling_factors
        self.n = n
        self.selected_energy = None
        self.selected_scaling = None
        self.skf = StratifiedKFold(n_splits=n_splits)

    def fit_sample(self, X, y):
        self.skf.get_n_splits(X, y)

        best_score = -np.inf

        for energy in self.energies:
            for scaling in self.scaling_factors:
                scores = []

                for train_idx, test_idx in self.skf.split(X, y):
                    X_train, y_train = CCR(energy=energy, scaling=scaling, n=self.n).\
                        fit_sample(X[train_idx], y[train_idx])

                    classifier = self.classifier.fit(X_train, y_train)
                    predictions = classifier.predict(X[test_idx])
                    scores.append(self.measure(y[test_idx], predictions))

                score = np.mean(scores)

                if score > best_score:
                    self.selected_energy = energy
                    self.selected_scaling = scaling

                    best_score = score

        return CCR(energy=self.selected_energy, scaling=self.selected_scaling, n=self.n).fit_sample(X, y)
