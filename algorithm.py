
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score,  balanced_accuracy_score, roc_auc_score

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_crossover, get_mutation, get_sampling, get_termination, get_selection
from operator import itemgetter

import visualisation
import pandas as pd
import pickle
import os


def distance(x, y, p_norm=2):
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)


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
        self.radii = None
        self.appended = None

        self.set_distances()

    def set_distances(self):

        self._classes = np.unique(self._y)
        sizes = [sum(self._y == c) for c in self._classes]

        assert len(self._classes) == len(set(sizes)) == 2

        self.minority_class = self._classes[np.argmin(sizes)]
        self.majority_class = self._classes[np.argmax(sizes)]
        self._minority = self._X[self._y == self.minority_class]
        self._majority = self._X[self._y == self.majority_class]

        if self.n is None:
            self._n = len(self._majority) - len(self._minority)

        self._distances = np.zeros((len(self._minority), len(self._majority)))

        for i in range(len(self._minority)):
            for j in range(len(self._majority)):
                self._distances[i][j] = distance(self._minority[i], self._majority[j])

    # def fit_sample(self, energy=0.25):
    #
    #     if not isinstance(energy, np.ndarray):
    #         energy = np.array([energy for i in range(self._minority.shape[0])])
    #
    #     if np.sum(energy) == 0:
    #         return self._X, self._y
    #
    #     energy = energy * (self._X.shape[1] ** self.scaling)
    #
    #     self.radii = np.zeros(len(self._minority))
    #     translations = np.zeros(self._majority.shape)
    #
    #     majority = np.copy(self._majority)
    #     minority = np.copy(self._minority)
    #
    #     for i in range(len(minority)):
    #         minority_point = minority[i]
    #         remaining_energy = energy[i]
    #         r = 0.0
    #         sorted_distances = np.argsort(self._distances[i])
    #         current_majority = 0
    #
    #         while True:
    #             if current_majority == len(majority):
    #                 if current_majority == 0:
    #                     radius_change = remaining_energy / (current_majority + 1.0)
    #                 else:
    #                     radius_change = remaining_energy / current_majority
    #
    #                 r += radius_change
    #
    #                 break
    #
    #             radius_change = remaining_energy / (current_majority + 1.0)
    #
    #             if self._distances[i, sorted_distances[current_majority]] >= r + radius_change:
    #                 r += radius_change
    #
    #                 break
    #             else:
    #                 if current_majority == 0:
    #                     last_distance = 0.0
    #                 else:
    #                     last_distance = self._distances[i, sorted_distances[current_majority - 1]]
    #
    #                 radius_change = self._distances[i, sorted_distances[current_majority]] - last_distance
    #                 r += radius_change
    #                 remaining_energy -= radius_change * (current_majority + 1.0)
    #                 current_majority += 1
    #
    #         self.radii[i] = r
    #
    #         for j in range(current_majority):
    #             majority_point = majority[sorted_distances[j]]
    #             d = self._distances[i, sorted_distances[j]]
    #
    #             if d < 1e-20:
    #                 dif = (1e-6 * np.random.rand(len(majority_point)) + 1e-6) * \
    #                                   np.random.choice([-1.0, 1.0], len(majority_point))
    #                 majority_point += dif
    #                 d = distance(minority_point, majority_point)
    #
    #             translation = (r - d) / d * (majority_point - minority_point)
    #
    #             translations[sorted_distances[j]] += translation
    #
    #     majority += translations
    #
    #     self.appended = []
    #
    #     for i in range(len(minority)):
    #         minority_point = minority[i]
    #         synthetic_samples = int(np.round(1.0 / (self.radii[i] * np.sum(1.0 / self.radii)) * self._n))
    #         r = self.radii[i]
    #
    #         for _ in range(synthetic_samples):
    #             self.appended.append(minority_point + taxicab_sample(len(minority_point), r))
    #
    #     self.appended = np.array(self.appended)
    #
    #     return np.concatenate([majority, minority, self.appended]), \
    #            np.concatenate([np.tile([self._majority_class], len(majority)),
    #                            np.tile([self._minority_class], len(minority) + len(self.appended))])

    def fit_sample(self, rads=0.25, fracs=1):

        if not isinstance(rads, np.ndarray):
            rads = np.array([rads for i in range(self._minority.shape[0])])

        if not isinstance(fracs, np.ndarray):
            fracs = np.array([fracs for i in range(self._minority.shape[0])])

        fracs = fracs/sum(fracs)

        self.radii = rads

        majority = np.copy(self._majority)
        minority = np.copy(self._minority)

        majority_to_delete = []

        for i in range(len(minority)):
            sorted_distances = np.argsort(self._distances[i])
            j = 0
            while j < len(sorted_distances) and self._distances[i][sorted_distances[j]] < rads[i]:
                majority_to_delete.append(sorted_distances[j])
                j += 1

        majority_to_delete = list(set(majority_to_delete))

        self.deleted_samples = majority[majority_to_delete]
        majority = np.delete(majority, majority_to_delete, axis=0)

        n = len(majority) - len(minority)

        self.appended = []
        self.samples = []
        for i in range(len(minority)):
            minority_point = minority[i]
            synthetic_samples = int(fracs[i] * n)
            self.samples.append(synthetic_samples)
            r = self.radii[i]

            for _ in range(synthetic_samples):
                self.appended.append(minority_point + taxicab_sample(len(minority_point), r))

        self.appended = np.array(self.appended)

        if len(self.appended) == 0:
            return np.concatenate([majority, minority]), \
               np.concatenate([np.full(len(majority), self.majority_class),
                               np.full( len(minority), self.minority_class)])

        return np.concatenate([majority, minority, self.appended]), \
               np.concatenate([np.full(len(majority), self.majority_class),
                               np.full( len(minority) + len(self.appended), self.minority_class)])

    # def fit_sample(self, samples=1):
    #
    #     if not isinstance(samples, np.ndarray):
    #         samples = np.array([samples for i in range(self._minority.shape[0])])
    #
    #     majority = np.copy(self._majority)
    #     minority = np.copy(self._minority)
    #     altered_labels = []
    #
    #     for i in range(len(minority)):
    #         minority_point = minority[i]
    #         sorted_distances = np.argsort(self._distances[i])
    #         try:
    #             if samples[i] > 0:
    #                 altered_labels.append(sorted_distances[:int(samples[i])])
    #         except Exception as e:
    #             print("o chuj chodzi")
    #             print(e)
    #
    #     if len(altered_labels) > 1:
    #         altered_labels = np.unique(np.concatenate(altered_labels))
    #
    #         self.appended = majority[altered_labels]
    #         majority = np.delete(majority, altered_labels, axis=0)
    #     else:
    #         self.appended = np.array([])
    #     if len(majority) > 0 and len(self.appended) > 0:
    #         return np.concatenate([majority, minority, self.appended]), \
    #                    np.concatenate([np.tile([self.majority_class], len(majority)).flatten(),
    #                                    np.tile([self.minority_class], len(minority) + len(self.appended)).flatten()])
    #     elif len(majority) == 0:
    #         return np.concatenate([minority, self.appended]), \
    #                                np.tile([self.minority_class], len(minority) + len(self.appended))
    #     else:
    #         return self._X, self._y
    #
    # def fit_sample_balanced(self, samples=1):
    #
    #     if not isinstance(samples, np.ndarray):
    #         samples = np.array([samples for i in range(self._minority.shape[0])])
    #
    #     if sum(samples) > 0:
    #         samples = samples/sum(samples)
    #
    #     majority = np.copy(self._majority)
    #     minority = np.copy(self._minority)
    #     altered_labels = []
    #     n = self._n/2
    #     self.samples = np.zeros((len(minority),))
    #
    #     for i in range(len(minority)):
    #         minority_point = minority[i]
    #         sorted_distances = np.argsort(self._distances[i])
    #         if len(altered_labels) > 0:
    #             sorted_distances = np.delete(sorted_distances, np.concatenate(altered_labels))
    #         change_no = int(samples[i]*n)
    #         self.samples[i] = change_no
    #         try:
    #             if samples[i] > 0:
    #                 altered_labels.append(sorted_distances[:change_no])
    #         except Exception as e:
    #             print("o chuj chodzi")
    #             print(e)
    #
    #     if len(altered_labels) > 1:
    #         altered_labels = np.unique(np.concatenate(altered_labels))
    #
    #         self.appended = majority[altered_labels]
    #         majority = np.delete(majority, altered_labels, axis=0)
    #     else:
    #         self.appended = np.array([])
    #     if len(majority) > 0 and len(self.appended) > 0:
    #         return np.concatenate([majority, minority, self.appended]), \
    #                np.concatenate([np.tile([self.majority_class], len(majority)),
    #                                np.tile([self.minority_class], len(minority) + len(self.appended))])
    #     elif len(majority) == 0:
    #         return np.concatenate([minority, self.appended]), \
    #                np.tile([self.minority_class], len(minority) + len(self.appended))
    #     else:
    #         return self._X, self._y



class PymooProblem(ElementwiseProblem):
    def __init__(self, n_min, aeccr, classifier, X_train, y_train, X_test, y_test, measures):
        self.n_var = 2*n_min
        # self.n_var = n_min
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
                         xl=np.full((self.n_var,), 0.0),
                         xu=np.full((self.n_var,), 1.0),
                         # xl=np.full((n_min,), 0),
                         # xu=np.full((n_min,), len(y_train)-n_min),
                         type_var=float)

    def _evaluate(self, x, out, *args, **kwargs):
        rads = x[0:int(self.n_var/2)]
        fracs = x[int(self.n_var/2):]

        new_X, new_y = self.aeccr.fit_sample(rads, fracs)
        # new_X, new_y = self.aeccr.fit_sample(x)
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


class SingleProblem(ElementwiseProblem):
    def __init__(self, n_min, aeccr, classifier, X_train, y_train, X_test, y_test):
        self.n_var = 2*n_min
        self.classifier = classifier
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.aeccr = aeccr
        super().__init__(n_var=self.n_var,
                         n_obj=1,
                         n_constr=0,
                         # xl=np.full((n_min,), 0.0),
                         # xu=np.full((n_min,), 0.1))
                         xl=0,
                         xu=np.concatenate((np.full((n_min,), 0.5), np.full((n_min,), 1.0))),
                         type_var=float)

    def _evaluate(self, x, out, *args, **kwargs):
        rads = x[0:int(self.n_var/2)]
        fracs = x[int(self.n_var/2):]
        new_X, new_y = self.aeccr.fit_sample(rads, fracs)
        c = clone(self.classifier)
        try:
            c.fit(new_X, new_y)
            y_pred = c.predict(self.X_test)
            out["F"] = -balanced_accuracy_score(self.y_test, y_pred)
        except:
            # print(x)
            # df_energy = pd.DataFrame(x)
            # df_X = pd.DataFrame(self.X_train)
            # df_y = pd.DataFrame(self.y_train)
            # df_energy.to_csv('blad_energy.csv')
            # df_X.to_csv('blad_X.csv')
            # df_y.to_csv('blad_y.csv')
            out["F"] = 0.0


class SingleProblemBalanced(ElementwiseProblem):
    def __init__(self, n_min, aeccr, classifier, X_train, y_train, X_test, y_test):
        self.n_var = n_min
        self.classifier = classifier
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.aeccr = aeccr
        super().__init__(n_var=self.n_var,
                         n_obj=1,
                         n_constr=0,
                         xl=np.full((2*n_min,), 0.0),
                         xu=np.full((2*n_min,), 1.0))

    def _evaluate(self, x, out, *args, **kwargs):
        new_X, new_y = self.aeccr.fit_sample(x)
        c = clone(self.classifier)
        try:
            c.fit(new_X, new_y)
            y_pred = c.predict(self.X_test)
            out["F"] = -balanced_accuracy_score(self.y_test, y_pred)
        except:
            # print(x)
            # df_energy = pd.DataFrame(x)
            # df_X = pd.DataFrame(self.X_train)
            # df_y = pd.DataFrame(self.y_train)
            # df_energy.to_csv('blad_energy.csv')
            # df_X.to_csv('blad_X.csv')
            # df_y.to_csv('blad_y.csv')
            out["F"] = 0.0


class MOO_CCRSelection:
    def __init__(self, classifier, measures, n_splits=1, energies=(0.25,), scaling_factors=(0.0,), n=None, criteria=['best'], test_size=0.2, save_directory='exp'):
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
        self.test_size = test_size

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

    def fit_sample(self, X, y, if_visualize=False):

        classes = np.unique(y)

        X = X.astype('float32')
        y = y.astype('float32')

        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size)
            sss.get_n_splits(X, y)
            for train_idx, test_idx in sss.split(X,y):
                pass
        except:
            train_idx = range(len(y))
            test_idx = range(len(y))

        sizes = [sum(y[train_idx] == c) for c in classes]
        min_size = min(sizes)
        aeccr = AECCR(X[train_idx], y[train_idx])
        problem = PymooProblem(min_size, aeccr, self.classifier, X[train_idx], y[train_idx], X[test_idx], y[test_idx], self.measures)
        algorithm = NSGA2(
            pop_size=100,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_ux"),
            mutation=get_mutation("real_pm"),
            eliminate_duplicates=True
        )
        # algorithm = NSGA2(
        #     pop_size=100,
        #     sampling=get_sampling("int_random"),
        #     crossover=get_crossover("int_sbx"),
        #     mutation=get_mutation("int_pm"),
        #     eliminate_duplicates=True
        # )
        termination = get_termination("n_gen", 200)
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=True,
                       verbose=True)

        if self.test_size == 0.5:
            sizes2 = [sum(y[test_idx] == c) for c in classes]
            min_size2 = min(sizes2)
            aeccr2 = AECCR(X[test_idx], y[test_idx])

            # if balanced:
            problem = PymooProblem(min_size2, aeccr2, self.classifier, X[test_idx], y[test_idx], X[train_idx],
                                   y[train_idx], self.measures)
            algorithm = NSGA2(
                pop_size=100,
                sampling=get_sampling("real_random"),
                crossover=get_crossover("real_ux"),
                mutation=get_mutation("real_pm"),
                eliminate_duplicates=True
            )
            termination = get_termination("n_gen", 200)
            res2 = minimize(problem,
                            algorithm,
                            termination,
                            seed=1,
                            save_history=True,
                            verbose=True)

            solutions = self.pick_solutions(res, self.criteria)
            solutions2 = self.pick_solutions(res2, self.criteria)

            for i in range(len(solutions)):
                solutions[i] = np.concatenate(
                    (solutions[i][0:min_size], solutions2[i][0:min_size2], solutions[i][min_size:], solutions2[i][min_size2:]))
        else:
            solutions = self.pick_solutions(res, self.criteria)

        if self.test_size == 0.5:
            idx = np.concatenate((train_idx, test_idx), axis=0)
            min_size += min_size2
        else:
            idx = train_idx

        cr = ['best_pre', 'best_rec', 'balanced']

        for ic, c in enumerate(cr):
            df_solutions = pd.DataFrame(solutions[ic])
            df_solutions.to_csv(os.path.join(self.save_directory, "solutions_{}.csv".format(c)))

        sizes = [sum(y[idx] == c) for c in classes]
        min_size = min(sizes)

        df_data = pd.DataFrame(X[idx][y[idx] == classes[np.argmin(sizes)]])
        df_data.to_csv(os.path.join(self.save_directory, "minority_class.csv"))

        aeccr_full = AECCR(np.concatenate((X[train_idx], X[test_idx]), axis=0),
                           np.concatenate((y[train_idx], y[test_idx]), axis=0))

        if if_visualize:
            print(self.save_directory)
            visualisation.visualize(X, y, classes[np.argmin(sizes)], file_name=self.save_directory)

            for i, s in enumerate(solutions):
                print(s)
                # X_, y_ = aeccr_full.fit_sample(s[0:min1+min2], s[min1+min2:])
                X_, y_ = aeccr_full.fit_sample(s[:min_size], s[min_size:])
                # visualisation.visualize(X_[:X.shape[0]], y_[:y.shape[0]], appended=aeccr_full.appended,
                #                         radii=aeccr_full.radii, file_name=os.path.join(self.save_directory, cr[i]),
                #                         samples=aeccr_full.samples, translated=aeccr_full.translated_samples)
                visualisation.visualize(np.concatenate((X[train_idx],X[test_idx]), axis=0), np.concatenate((y[train_idx],y[test_idx]), axis=0),
                                        aeccr_full.minority_class, aeccr_full.majority_class, appended=aeccr_full.appended,
                                        file_name=os.path.join(self.save_directory, cr[i]), samples=aeccr_full.samples)

        return [aeccr_full.fit_sample(x[0:min_size], x[min_size:]) for x in solutions]
        # return [aeccr_full.fit_sample(x) for x in solutions_merged]

class not_MOO_CCRSelection:
    def __init__(self, classifier, n_splits=1, energies=(0.25,), scaling_factors=(0.0,), n=None, test_size=0.2, save_directory='exp'):
        self.classifier = classifier
        self.n_splits = n_splits
        self.energies = energies
        self.scaling_factors = scaling_factors
        self.n = n
        self.save_directory = save_directory
        self.selected_energy = None
        self.selected_scaling = None
        self.test_size = test_size

    @staticmethod
    def binary_tournament(pop, P, **kwargs):

        # The P input defines the tournaments and competitors
        n_tournaments, n_competitors = P.shape

        if n_competitors != 2:
            raise Exception("Only pressure=2 allowed for binary tournament!")

        # the result this function returns
        import numpy as np
        S = np.full(n_tournaments, -1, dtype=np.int)

        # now do all the tournaments
        for i in range(n_tournaments):
            a, b = P[i]

            # if the first individiual is better, choose it
            if pop[a].F < pop[b].F:
                S[i] = a

            # otherwise take the other individual
            else:
                S[i] = b

        return S

    def fit_sample(self, X, y, if_visualize=False, balanced=False, transform=False):

        classes = np.unique(y)

        X = X.astype('float32')
        y = y.astype('float32')

        if if_visualize:
            X, y = visualisation.prepare_data(X,y)

        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size)
            sss.get_n_splits(X, y)
            for train_idx, test_idx in sss.split(X,y):
                pass
        except:
            train_idx = range(len(y))
            test_idx = range(len(y))

        sizes = [sum(y[train_idx] == c) for c in classes]
        min_size = min(sizes)

        aeccr = AECCR(X[train_idx], y[train_idx])

        # if balanced:
        problem = SingleProblem(min_size, aeccr, self.classifier, X[train_idx], y[train_idx], X[test_idx],
                                y[test_idx])
        algorithm = GA(
            pop_size=100,
            sampling=get_sampling("real_random"),
            selection=get_selection('tournament', self.binary_tournament),
            crossover=get_crossover("real_ux"),
            mutation=get_mutation("real_pm"),
            eliminate_duplicates=True
        )
        # else:
        # problem = SingleProblem(min_size, aeccr, self.classifier, X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        # algorithm = GA(
        #     pop_size=100,
        #     sampling=get_sampling("int_random"),
        #     selection=get_selection('tournament', self.binary_tournament),
        #     crossover=get_crossover("int_sbx"),
        #     mutation=get_mutation("int_pm"),
        #     eliminate_duplicates=True
        # )
        termination = get_termination("n_gen", 200)
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=True,
                       verbose=True)

        if self.test_size == 0.5:
            sizes2 = [sum(y[test_idx] == c) for c in classes]
            min_size2 = min(sizes2)
            aeccr2 = AECCR(X[test_idx], y[test_idx])

            # if balanced:
            problem = SingleProblem(min_size2, aeccr2, self.classifier, X[test_idx], y[test_idx], X[train_idx],
                                    y[train_idx])
            algorithm = GA(
                pop_size=100,
                sampling=get_sampling("real_random"),
                selection=get_selection('tournament', self.binary_tournament),
                crossover=get_crossover("real_ux"),
                mutation=get_mutation("real_pm"),
                eliminate_duplicates=True
            )
            # else:
            #     problem = SingleProblem(min_size2, aeccr2, self.classifier, X[test_idx], y[test_idx], X[train_idx], y[train_idx])
            #     algorithm = GA(
            #         pop_size=100,
            #         sampling=get_sampling("int_random"),
            #         selection=get_selection('tournament', self.binary_tournament),
            #         crossover=get_crossover("int_sbx"),
            #         mutation=get_mutation("int_pm"),
            #         eliminate_duplicates=True
            #     )
            termination = get_termination("n_gen", 200)
            res2 = minimize(problem,
                           algorithm,
                           termination,
                           seed=1,
                           save_history=True,
                           verbose=True)

            solutions = res.X
            solutions2 = res2.X


            # solutions_merged = [np.zeros((min1+min2)) for sol in solutions]
            solutions = np.concatenate((solutions[0:min_size], solutions2[0:min_size2], solutions[min_size:], solutions2[min_size2:]))
        else:
            solutions = res.X


        # for i in range(len(solutions)):
        #     solutions_merged[i][:min1] = solutions[i][:min1]
        #     solutions_merged[i][min1:] = solutions2[i][:]
        if self.test_size == 0.5:
            idx = np.concatenate((train_idx,test_idx), axis=0)
            min_size += min_size2
        else:
            idx = train_idx

        aeccr_full = AECCR(X[idx], y[idx])

        df_solutions = pd.DataFrame(solutions)
        df_solutions.to_csv(os.path.join(self.save_directory, "solutions.csv"))

        df_data = pd.DataFrame(X[idx][y[idx] == aeccr_full.minority_class])
        df_data.to_csv(os.path.join(self.save_directory, "minority_class.csv"))

        # if balanced:
        #     newX, newy = aeccr_full.fit_sample_balanced(solutions)
        # else:
        newX, newy = aeccr_full.fit_sample(solutions[:min_size], solutions[min_size:])

        if transform:
            df_set = pd.DataFrame(data=newX)
            df_set['label'] = newy

            df_set.to_csv(os.path.join(self.save_directory, "transformed_data.csv"))

        if if_visualize:
            print(self.save_directory)
            visualisation.visualize(X, y, classes[np.argmin(sizes)], file_name=self.save_directory)
            # X_, y_ = aeccr_full.fit_sample(s[0:min1+min2], s[min1+min2:])
            # X_, y_ = aeccr_full.fit_sample(solutions)
            # visualisation.visualize(X_[:X.shape[0]], y_[:y.shape[0]], appended=aeccr_full.appended,
            #                         radii=aeccr_full.radii, file_name=os.path.join(self.save_directory, cr[i]),
            #                         samples=aeccr_full.samples, translated=aeccr_full.translated_samples)
            visualisation.visualize(X[idx], y[idx],
                                    aeccr_full.minority_class, appended=aeccr_full.appended,
                                    file_name=os.path.join(self.save_directory, 'result'), samples=aeccr_full.samples, radii=aeccr_full.radii,
                                    deleted_samples=aeccr_full.deleted_samples)

        # return [aeccr_full.fit_sample(x[0:min1+min2], x[min1+min2:]) for x in solutions_merged]
        return newX, newy

class CCR:
    def __init__(self, energy=0.25, scaling=0.0, n=None):
        self.energy = energy
        self.scaling = scaling
        self.n = n
        self.radii = None
        self.appended = None

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

        self.radii = np.zeros(len(minority))
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

            self.radii[i] = r

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

        self.appended = []

        for i in range(len(minority)):
            minority_point = minority[i]
            synthetic_samples = int(np.round(1.0 / (self.radii[i] * np.sum(1.0 / self.radii)) * n))
            r = self.radii[i]

            for _ in range(synthetic_samples):
                self.appended.append(minority_point + taxicab_sample(len(minority_point), r))

        self.appended = np.array(self.appended)

        return np.concatenate([majority, minority, self.appended]), \
               np.concatenate([np.tile([majority_class], len(majority)),
                               np.tile([minority_class], len(minority) + len(self.appended))])


class CCRSelection:
    def __init__(self, classifier, measure, n_splits=5, energies=(0.25,), scaling_factors=(0.0,), n=None, save_directory='exp'):
        self.classifier = classifier
        self.measure = measure
        self.n_splits = n_splits
        self.energies = energies
        self.scaling_factors = scaling_factors
        self.n = n
        self.save_directory = save_directory
        self.selected_energy = None
        self.selected_scaling = None
        self.skf = StratifiedKFold(n_splits=n_splits)

    def fit_sample(self, X, y, if_visualize=False):
        self.skf.get_n_splits(X, y)

        best_score = -np.inf

        if if_visualize:
            X, y = visualisation.prepare_data(X, y)

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

        ccr = CCR(energy=self.selected_energy, scaling=self.selected_scaling, n=self.n)
        X_, y_ = ccr.fit_sample(X, y)
        # visualisation.visualize(X_[:X.shape[0]], y_[:y.shape[0]], appended=ccr.appended, radii=ccr.radii, file_name=self.save_directory)
        print(self.save_directory)
        print(self.selected_energy)

        return CCR(energy=self.selected_energy, scaling=self.selected_scaling, n=self.n).fit_sample(X, y)
