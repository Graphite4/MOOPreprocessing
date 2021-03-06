from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score,  balanced_accuracy_score, roc_auc_score


from imblearn.over_sampling import SMOTE, RandomOverSampler
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np
import algorithm
import os
import argparse

from sklearn.base import clone
from DatasetsCollection import load


metrics = [balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]
criteria = ['best_precision', 'best_recall', 'balanced']
algorithms = ['new_CCR_best_precision', 'new_CCR_best_recall', 'new_CCR_balanced', 'old_CCR', 'SMOTE', "random_over"]

file_list = [ "ecoli-0-1-3-7_vs_2-6", "glass-0-1-6_vs_2", "glass-0-1-6_vs_5", "glass2",
             "glass4", "glass5", "page-blocks-1-3_vs_4", "yeast-0-5-6-7-9_vs_4", "yeast-1-2-8-9_vs_7",
             "yeast-1-4-5-8_vs_7", "yeast-1_vs_7", "yeast-2_vs_4", "yeast-2_vs_8", "yeast4", "yeast5", "yeast6",
             "cleveland-0_vs_4", "ecoli-0-1-4-7_vs_2-3-5-6",
              "ecoli-0-1_vs_2-3-5", "ecoli-0-2-6-7_vs_3-5",
             "ecoli-0-6-7_vs_3-5", "ecoli-0-6-7_vs_5", "glass-0-1-4-6_vs_2", "glass-0-1-5_vs_2",
             "yeast-0-2-5-6_vs_3-7-8-9", "yeast-0-3-5-9_vs_7-8",
             "abalone-17_vs_7-8-9-10", "abalone-19_vs_10-11-12-13",
             "abalone-20_vs_8-9-10", "abalone-21_vs_8", "flare-F", "kddcup-buffer_overflow_vs_back",
             "kddcup-rootkit-imap_vs_back", "kr-vs-k-zero_vs_eight", "poker-8-9_vs_5", "poker-8-9_vs_6", "poker-8_vs_6",
             "poker-9_vs_7", "winequality-red-3_vs_5", "winequality-red-4",
             "winequality-red-8_vs_6-7",
             "winequality-red-8_vs_6", "winequality-white-3-9_vs_5", "winequality-white-3_vs_7",
             "winequality-white-9_vs_4", "zoo-3", "ecoli1", "ecoli2", "ecoli3", "glass0", "glass1", "haberman",
             "page-blocks0", "pima", "vehicle1", "vehicle3", "yeast1", "yeast3", "abalone19",
             "abalone9-18"
]
# file_list = ["page-blocks0", "pima", "vehicle1", "vehicle3", "yeast1", "yeast3"]
# file_list = ["pima", "yeast4", "yeast5", "winequality-white-3-9_vs_5"]


# def experiment(data, classifier, path="plot", **kwargs):
#     metrics_array = np.empty((len(criteria), 10, len(metrics)))
#     # metrics_array = np.empty((10, len(metrics)))
#     try:
#         os.mkdir(path)
#     except:
#         print("ojej")
#
#     for i, fold in enumerate(data):
#         fold_path = os.path.join(path, "fold" + str(i))
#         try:
#             os.mkdir(fold_path)
#         except:
#             print("ojej")
#         # #
#         ccr = algorithm.MOO_CCRSelection(classifier, [precision_score, recall_score], criteria=['best','balanced'], test_size=0.5, save_directory=fold_path, )
#         resampled_data = ccr.fit_sample(fold[0][0], fold[0][1], if_visualize=True)
#         # ccr = algorithm.CCRSelection(classifier, balanced_accuracy_score, save_directory=fold_path)
#         # resampled_data = ccr.fit_sample(fold[0][0], fold[0][1], if_visualize=True)
#         # for ic, cr in enumerate(criteria):
#         #     data = resampled_data[ic]
#         #     c = clone(classifier)
#         #     c.fit(data[0], data[1])
#         #     y_pred = c.predict(fold[1][0])
#         #     df = pd.DataFrame(y_pred)
#         #     df.to_csv(os.path.join(fold_path, "classifier_prediction_" + cr + ".csv"))
#         #     for im, metric in enumerate(metrics):
#         #         metrics_array[ic, i, im] = metric(fold[1][1], y_pred)
#
#         # data, y = resampled_data
#         # c = clone(classifier)
#         # c.fit(data, y)
#         # # y_pred = c.predict(fold[1][0])
#         # # df = pd.DataFrame(y_pred)
#         # # df.to_csv(os.path.join(fold_path, "classifier_prediction_ccr.csv"))
#         # # for im, metric in enumerate(metrics):
#         # #     metrics_array[i, im] = metric(fold[1][1], y_pred)
#
#     return np.mean(metrics_array, axis=1)
#     # return np.mean(metrics_array, axis=0)
def experiment_parallel(data,fold_no, classifier, path="plot", test_size=0.5, balanced=False, transformed=False, **kwargs):
    metrics_array = np.zeros((len(algorithms), 10, len(metrics)))
    try:
        os.mkdir(path)
    except:
        print("ojej")
    print(path)

    fold = data[fold_no]
    i = fold_no

    print("Fold {}".format(str(i)))
    fold_path = os.path.join(path, "fold" + str(i))
    try:
        os.mkdir(fold_path)
    except:
        print("ojej")
    # try:
    # not_moo_ccr = algorithm.not_MOO_CCRSelection(classifier, test_size=test_size, save_directory=fold_path, )
    # resampled_data_from_not_moo_ccr = not_moo_ccr.fit_sample(fold[0][0], fold[0][1], if_visualize=False,
    #                                                              balanced=balanced, transform=transformed)
    moo_ccr = algorithm.MOO_CCRSelection(classifier, test_size=test_size, save_directory=fold_path,
                                         measures=[precision_score, recall_score], criteria=['best', 'balanced'])
    resampled_data_from_moo_ccr = moo_ccr.fit_sample(fold[0][0], fold[0][1])
    # except Exception as e:
    #     resampled_data_from_not_moo_ccr = None
    #     print('not_MOO_CCR')
    #     print(e)
    try:
        energies = (0.1, 0.25, 0.5, 1, 5, 10, 50, 100)
        ccr = algorithm.CCRSelection(classifier, balanced_accuracy_score, energies=energies,
                                     save_directory=fold_path)
        resampled_data_from_ccr = [ccr.fit_sample(fold[0][0], fold[0][1], if_visualize=False)]
    except Exception as e:
        resampled_data_from_ccr = None
        print('CCR')
        print(e)
    try:
        smote = SMOTE()
        resampled_data_from_smote = [smote.fit_resample(fold[0][0], fold[0][1])]
    except Exception as e:
        resampled_data_from_smote = None
        print('SMOTE')
        print(e)
    try:
        random = RandomOverSampler()
        resampled_data_from_random = [random.fit_resample(fold[0][0], fold[0][1])]
    except Exception as e:
        resampled_data_from_random = None
        print('Random Oversampling')
        print(e)
    resampled_data = resampled_data_from_moo_ccr + resampled_data_from_ccr + resampled_data_from_smote + resampled_data_from_random
    for ia in range(len(algorithms)):
        data = resampled_data[ia]
        if data is not None:
            c = clone(classifier)
            c.fit(data[0], data[1])
            y_pred = c.predict(fold[1][0])
            df = pd.DataFrame(y_pred)
            df.to_csv(os.path.join(fold_path, "classifier_prediction_" + algorithms[ia] + ".csv"))
            # for im, metric in enumerate(metrics):
            #     metrics_array[ia, i, im] = metric(fold[1][1], y_pred)
    return


def experiment(data, classifier, path="plot", test_size=0.5, balanced=False, transformed=False, **kwargs):
    metrics_array = np.zeros((len(algorithms), 10, len(metrics)))
    try:
        os.mkdir(path)
    except:
        print("ojej")
    print(path)

    for i, fold in enumerate(data):
        print("Fold {}".format(str(i)))
        fold_path = os.path.join(path, "fold" + str(i))
        try:
            os.mkdir(fold_path)
        except:
            print("ojej")
        # try
        # except Exception as e:
        #     resampled_data_from_not_moo_ccr = None
        #     print('not_MOO_CCR')
        #     print(e)
        try:
            energies = (0.1, 0.25, 0.5, 1, 5, 10, 50, 100)
            ccr = algorithm.CCRSelection(classifier, balanced_accuracy_score, energies=energies, save_directory=fold_path)
            resampled_data_from_ccr = ccr.fit_sample(fold[0][0], fold[0][1], if_visualize=True)
        except Exception as e:
            resampled_data_from_ccr = None
            print('CCR')
            print(e)
        try:
            smote = SMOTE()
            resampled_data_from_smote = smote.fit_resample(fold[0][0], fold[0][1])
        except Exception as e:
            resampled_data_from_smote = None
            print('SMOTE')
            print(e)
        try:
            random = RandomOverSampler()
            resampled_data_from_random = random.fit_resample(fold[0][0], fold[0][1])
        except Exception as e:
            resampled_data_from_random = None
            print('Random Oversampling')
            print(e)
        resampled_data = [resampled_data_from_not_moo_ccr, resampled_data_from_ccr, resampled_data_from_smote, resampled_data_from_random]
        for ia in range(len(algorithms)):
            data = resampled_data[ia]
            if data is not None:
                c = clone(classifier)
                c.fit(data[0], data[1])
                y_pred = c.predict(fold[1][0])
                df = pd.DataFrame(y_pred)
                df.to_csv(os.path.join(fold_path, "classifier_prediction_" + algorithms[ia] + ".csv"))
                for im, metric in enumerate(metrics):
                    metrics_array[ia, i, im] = metric(fold[1][1], y_pred)


    return np.mean(metrics_array, axis=1)

def conduct_experiment(path="experiments-no-radii-no-bound-full3", dataset='all',fold=10, classifier_type='DT', test_size=0.5, balanced=False, transformed=False):
    # data_set = []
    # for file in file_list:
    #     data_set.append(load(file))
    #
    # try:
    #     os.mkdir(path)
    # except:
    #     print("ojej")
    #
    # classifier = DecisionTreeClassifier(random_state=7)
    # metrics_dfs = [pd.DataFrame(index=file_list, columns=criteria) for metric in metrics]
    # for i, data in enumerate(data_set):
    #     results = experiment(data, classifier, os.path.join(path, file_list[i]))
    #     for im, metric in enumerate(metrics):
    #         metrics_dfs[im].at[file_list[i]] = results[:,im]
    #         # metrics_dfs[im].at[file_list[i]] = results[im]
    #         metrics_dfs[im].to_csv(os.path.join(path, "results_" + metric.__name__ + ".csv"))

    data_set = []
    if dataset == 'all':
        names = file_list
        for file in file_list:
            data_set.append(load(file, transformed=transformed))

    else:
        names = [dataset]
        data_set.append(load(dataset))

    try:
        os.mkdir(path)
    except:
        print("ojej")

    if classifier_type == 'DT':
        classifier = DecisionTreeClassifier(random_state=7)
    elif classifier_type == 'SVM':
        classifier = SVC(random_state=13)
    elif classifier_type == 'knn':
        classifier = KNeighborsClassifier()
    elif classifier_type == 'Bayes':
        classifier = GaussianNB()
    elif classifier_type == "MLP":
        classifier = MLPClassifier(random_state=42, max_iter=500)
        
    metrics_dfs = [pd.DataFrame(index=names, columns=algorithms) for metric in metrics]
    for metric in metrics:
        try:
            os.mkdir(os.path.join(path,metric.__name__))
        except:
            pass
    try:
        os.mkdir(os.path.join(path,"results"))
    except:
        pass

    for i, data in enumerate(data_set):
        if fold < 10:
            experiment_parallel(data, fold, classifier, os.path.join(path,"results", names[i]), test_size, balanced)
        else:
            results = experiment(data, classifier, os.path.join(path,"results", names[i]), test_size, balanced)
            for im, metric in enumerate(metrics):
                metrics_dfs[im].at[names[i]] = results[:, im]
                # metrics_dfs[im].at[file_list[i]] = results[im]
                metrics_dfs[im].to_csv(os.path.join(path,metric.__name__, "results_" + dataset + ".csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--fold", type=int, default=10)
    parser.add_argument("--classifier", type=str)
    parser.add_argument("--test_size", type=float)
    parser.add_argument("--balanced", dest='balanced', action='store_true')
    parser.add_argument("--unbound", dest='balanced', action="store_false")
    parser.set_defaults(balanced=False)
    parser.add_argument("--transformed", dest='transformed', action='store_true')
    parser.set_defaults(transformed=False)
    args = parser.parse_args()
    conduct_experiment(args.result_directory, args.dataset,args.fold, args.classifier, args.test_size, args.balanced, args.transformed)



