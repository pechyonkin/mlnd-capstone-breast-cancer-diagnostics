# Data handling and analysis
from scipy import stats
import numpy as np
import pandas as pd

# used for plotting confustion matrix
import itertools
import matplotlib.pyplot as plt

# for data preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA

# custom scorers
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, make_scorer, confusion_matrix

# for model selection and cross-validation
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit

# importing all classifiers needed
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# pipelining
from sklearn.pipeline import Pipeline

# for saving to a table
from code import tablemaker as tb


def make_data(path_to_csv="./data/breast-cancer-kaggle.csv", verbose=True):
    ''' returns a tuple:
    (DataFrame of feature values, Series of string target labels) '''
    data = pd.read_csv(path_to_csv)

    X = data[data.columns[2:]]
    y = data[data.columns[1]]
    
    if verbose:
        print "-"*80
        print "MAKING DATA"
        print "-"*80
        print "Shape of data is: {}".format(data.shape)
        n_malignant = sum(data['diagnosis'] == "M")
        n_benign = sum(data['diagnosis'] == "B")
        n_total = data.shape[0]
        print "-"*80
        print "Classes add up normally: {}".format(n_malignant + n_benign == n_total)
        print "Malignant class: {:.2f}".format(n_malignant/float(n_total))
        print "Benign class: {:.2f}".format(n_benign/float(n_total))
        print "-"*80
        print("Printing all feature names:")
        for i,f in enumerate(X.columns):
            print i, f
        print "-"*80
    
    return (X, y)


def encode_labels(y, verbosity=True):
    ''' encodes string labels to integers
        returns coded labels and fit LabelEncoder object
        '''
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    y_coded = pd.Series(label_encoder.transform(y))
    if verbosity:
        print "Checking if lenghts are equal: {}".format(len(y) == len(y_coded))
        print "Label encoder classes:", label_encoder.classes_
        print "Inverse transforming [0,1]:", label_encoder.inverse_transform([0,1])
    return (y_coded, label_encoder)


def remove_outliers(X, y, iqr_threshold, verbose=True):
    ''' removes data points that are outside
        given multiple of IQRs from median
        and returns data and labels without outliers '''
    # calculates row index list without outliers
    indices_without_outliers = (np.abs((X - np.median(X, axis=0)) / stats.iqr(X, axis=0)) < float(iqr_threshold)).all(axis=1)
    
    # selects only non-outlier rows
    new_X = X[indices_without_outliers]
    new_y = y[indices_without_outliers]
    
    if verbose:
        print "-"*80
        print "REMOVING OUTLIERS"
        print "-"*80
        print "There were {} data points before removing outliers.".format(X.shape[0])
        print "There are {} data points after removing outliers.".format(new_X.shape[0])
        print "Outliers removed: {}".format(X.shape[0] - new_X.shape[0])
        print "-"*80
    
    return (new_X, new_y)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, 
                          verbose=True):
    ''' This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        ---
        This code is taken from:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def make_logspace(start=-3, end=3, base=10):
    ''' returns a logspace with steps equal to powers of 10
        used to make nice grids for SVMs
        '''
    n = end - start + 1
    space = np.logspace(start, end, num=n, base=base)
    return space


def get_clf_info(clf, verbose=True):
    ''' takes a classifier and returns a string of its class name
        if SVM then returns kernel in parentheses '''
    if clf.__class__.__name__ == 'SVC':
        return "{} ({})".format(clf.__class__.__name__, clf.kernel)
    else:
        return clf.__class__.__name__
    
    
def get_tpr_tnr(confusion_matrix, verbose=True):
    ''' takes a confusion matrix object and returns floats:
        TPR: true positive rate (a.k.a. recall, sensitivity)
        TNR: true negative rate (a.k.a. specificity) '''
    tp = float(confusion_matrix[1,1])
    tn = float(confusion_matrix[0,0])
    fp = float(confusion_matrix[0,1])
    fn = float(confusion_matrix[1,0])
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return (tpr, tnr)


def make_df_from_results(results, classifiers, verbose=True):
    ''' takes a list in the form produced by
        basic_implementation_calculate_results()
        and makes a DataFrame with average values from it
        ---
        then this DataFrame can be used to make a .tex table
        for the final report'''
    result_df = pd.DataFrame()

    # make a table with results
    # columns: accuracy, sd of accuracy, tpr, tnr
    for i,e in enumerate(results):
        row = pd.Series([np.mean(e[1]),np.std(e[1]),np.mean(e[2]),np.mean(e[3])],name=get_clf_info(classifiers[i]),index=["Accuracy","SD","TPR","TNR"])
        result_df = result_df.append(row)

    result_df = result_df[["Accuracy","SD","TPR","TNR"]] # rearrange columns in desired order

    return result_df


def calculate_improvements(old_df, new_df, 
                          write_to_table=False, 
                          path_to_output="./tex/tables/",
                          scap="Table of differences", 
                          caption="Table of differences", 
                          label="default-diff", 
                          tabname="default-diff", 
                          precision=3, 
                          fcol="Classifier"):
    ''' calculate imrovements of new_df over old_df '''
    # old df has fewer columns and rows, make sure that
    # new df has only the same as the old has
    old_cols = old_df.columns
    old_index = old_df.index
    new_df = new_df[old_cols]
    new_df = new_df.loc[old_index]
    diff_df = (new_df - old_df) / old_df
    if write_to_table:
        tb.write_to_a_table(
            diff_df, 
            path_to_output=path_to_output,
            scap=scap,
            caption=caption,
            label=label,
            tabname=tabname,
            precision=precision,
            first_column_header=fcol)
    return diff_df


def make_list_of_classifiers(rand_state = 42, verbose=True):
    ''' returns a list of classifiers with default parametes
        classifiers are ones outlined in report, section of basic implementation
        ---
        the goal is to set a baseline performance and compare to benchmark '''
    # SVM
    clf_svm_lin = SVC(kernel='linear', random_state=rand_state)
    clf_svm_rbf = SVC(kernel='rbf', random_state=rand_state)
    clf_svm_quad = SVC(kernel='poly', degree=2, random_state=rand_state)
    # Ensembles
    clf_rand_forest = RandomForestClassifier(random_state=rand_state)
    clf_adaboost = AdaBoostClassifier(random_state=rand_state)
    # Naive Bayes
    clf_naive = GaussianNB()

    classifiers = [clf_svm_lin, clf_svm_rbf, clf_svm_quad, clf_rand_forest, clf_adaboost, clf_naive]
    
    if verbose:
        print "-"*80
        print "MAKING THE LIST OF CLASSIFIERS"
        print "-"*80
        print "Printing classifiers:"
        for clf in classifiers:
            print get_clf_info(clf)
        print "-"*80
            
    return classifiers


def basic_implementation_calculate_results(XX, yy, classifiers, splits=10, test_size=0.1, rand_state=42, verbose=True):
    ''' performs evaluation of untuned classifiers
        ---
        inputs:
            X: feature DataFrame
            y_coded: target labels in binary numerical format (0s and 1s)
        ---
        goes through each classifier in the list of instantiated classifiers
        uses cross-validation, calculates for each fold:
            accuracy, sd of accuracy, tpr, tnr
        ---
        output is a list of each classifier results (itself a list)
        ---
        results = [
                    [
                        'clf_1_name', 
                        [fold_1_acc, fold_2_acc, ...], 
                        [fold_1_tpr, fold_2_tpr, ...], 
                        [fold_1_tnr, fold_2_tnr, ...]
                    ],
                    ...
                  ]
        '''
    cv = StratifiedShuffleSplit(n_splits=splits, random_state=rand_state, test_size=test_size)
    results = []

    if verbose:
        print "-"*80
        print "PERFORMING CROSS-VALIDATION FOR BASIC IMPLEMENTATION"
        print "-"*80
    for clf in classifiers:
        ''' performs cross-validation
            and calculates metrics for
            each classifier in the list '''
        row = []
        row.append(get_clf_info(clf))
        clf_name = get_clf_info(clf)
        if verbose:
            print "{}".format(clf_name)
            print "-"*80
        accuracy_values = []
        tpr_values = []
        tnr_values = []
        for i, (train_index, test_index) in enumerate(cv.split(XX,yy)):
            if verbose: print "{}: fold {}".format(clf_name, i+1)
            # fit and predict
            clf.fit(XX.iloc[train_index], yy.iloc[train_index])
            yy_pred = clf.predict(XX.iloc[test_index])
            yy_true = yy.iloc[test_index]
            # calculate metrics
            cm = confusion_matrix(yy_true, yy_pred)
            tpr, tnr = get_tpr_tnr(cm)
            accuracy = accuracy_score(yy_true, yy_pred)
            accuracy_values.append(accuracy)
            tpr_values.append(tpr)
            tnr_values.append(tnr)
        row.append(accuracy_values)
        row.append(tpr_values)
        row.append(tnr_values)
        if verbose:
            print "Average accuracy is: {}".format(np.mean(accuracy_values))
            print "Average TPR is: {}".format(np.mean(tpr_values))
            print "Average TNR is: {}".format(np.mean(tnr_values))
            print "-" * 80
        results.append(row)
    if verbose:
        print "FINAL RESULTS\n" + "-"*80
        print "Classifier: accuracy (SD), TPR, TNR\n" + "-"*80
        for e in results:
            print "{}: accuracy {:.4f} ({:.4f}); TPR {:.4f}; TNR {:.4f}".format(e[0], np.mean(e[1]), np.std(e[1]), np.mean(e[2]), np.mean(e[3]))
        print "-"*80
    return results


def get_pipeline_param_grid(clf, dim_reducer=None):
    ''' returns a parameter grid for the pipeline
        only PCA supprted for now
        can be extended to others'''
    clf_name = get_clf_info(clf)
    print 'clf_name: {}'.format(clf_name)
#     dim_reducer_name = cs.get_clf_info(dim_reducer)
    
    # for support vector machines
    C_grid = make_logspace()
    # for ensemble methods
    n_estim = [10, 50, 100]
    # for PCA
    n_comp = [2, 5, 10, 15, 20, 25]
    
    clf_params_dict = {
        'SVC (linear)':{
            'C': C_grid
        },
        'SVC (rbf)':{
            'C': C_grid,
            'gamma': make_logspace(start=-9, end=3)
        },
        'SVC (poly)':{
            'C':C_grid,
            'coef0': np.linspace(-1,1,6)
        },
        'RandomForestClassifier':{
            'n_estimators': n_estim
        },
        'AdaBoostClassifier':{
            'n_estimators': n_estim
        },
        'GaussianNB':{}
    }
    
    result = [{
            'dim_reduce__n_components':n_comp
        }]
    
    for k in clf_params_dict[clf_name].keys():
        result[0]['classify__' + k] = clf_params_dict[clf_name][k]
        
    return result


def piper(X, y, clf, scorer=None, grid=None, dim_reducer=PCA, validation=StratifiedShuffleSplit, n_splits=3, test_size=0.1, verbose=1, rand=42):
    ''' performs grid search of pipeline, returns fitted GridSearchSV '''
    
    pipeline = Pipeline([
            ('min_max_scaler', MinMaxScaler(feature_range=(-1,1))), # good for SVMs
            ('dim_reduce', dim_reducer()), # names of steps should not be changed
            ('classify', clf) # other code depend on them, will break if change
        ])
    # construct Pipeline-compatible parameter grid with 2-steps
    if grid==None:
        param_grid = get_pipeline_param_grid(clf, dim_reducer)
    else:
        param_grid = grid
    
    # initialize CV object with given parameters
    cv_obj = validation(n_splits=n_splits, test_size=0.1, random_state=rand)
    # create the grid for cross validation
    grid = GridSearchCV(pipeline, cv=cv_obj, param_grid=param_grid, scoring=scorer, verbose=verbose)
    grid.fit(X, y)
    
    return grid


def improved_nested_calculate_results(XX, yy, classifiers, 
                                      scorer=None,
                                      outer_loop_splits=1,
                                      grid_search_loops=1, 
                                      test_size=0.2, 
                                      rand_state=42, 
                                      verbose=True):
    ''' performs nested cross-validation
        inner loop does grid search for best model
        outer loop estimates out-of-sample performance
        returns a tuple: (outer loop scores, GridSearchCV for each classifier)
        '''
    n = 52 # for printing lines

    results = []
    grids = [] # fitted girds; size = n_classifiers * n_outer_splits

    cv = StratifiedShuffleSplit(n_splits=outer_loop_splits, random_state=rand_state, test_size=test_size)

    if verbose:
        print "-"*n
        print "PERFORMING NESTED CROSS-VALIDATION WITH GRID_SEARCH_CV"
        print "-"*n
    for clf in classifiers:
        ''' performs cross-validation
            and calculates metrics for
            each classifier in the list '''
        row = []
        row.append(get_clf_info(clf))
        clf_name = get_clf_info(clf)
        if verbose:
            print "{}".format(clf_name)
            print "-"*n
        accuracy_values = []
        tpr_values = []
        tnr_values = []
        # this loop performs out-of-sample performance estimation of best models
        for i, (train_index, test_index) in enumerate(cv.split(XX,yy)):
            if verbose: print "{}: fold {}".format(clf_name, i+1)
            # fit and predict (model selection and grid search CV is performed by piper)
            gd = piper(XX.iloc[train_index], yy.iloc[train_index], clf, scorer=scorer, n_splits=grid_search_loops)
            grids.append(gd)
            yy_pred = gd.predict(XX.iloc[test_index])
            yy_true = yy.iloc[test_index]
            # calculate metrics
            cm = confusion_matrix(yy_true, yy_pred)
            tpr, tnr = get_tpr_tnr(cm)
            accuracy = accuracy_score(yy_true, yy_pred)
            accuracy_values.append(accuracy)
            tpr_values.append(tpr)
            tnr_values.append(tnr)
        row.append(accuracy_values)
        row.append(tpr_values)
        row.append(tnr_values)
        if verbose:
            print "Average accuracy is: {}".format(np.mean(accuracy_values))
            print "Average TPR is: {}".format(np.mean(tpr_values))
            print "Average TNR is: {}".format(np.mean(tnr_values))
            print "-" * n
        results.append(row)
    if verbose:
        print "FINAL RESULTS\n" + "-"*n
        print "Classifier: accuracy (SD), TPR, TNR\n" + "-"*n
        for e in results:
            print "{}: accuracy {:.4f} ({:.4f}); TPR {:.4f}; TNR {:.4f}".format(e[0], np.mean(e[1]), np.std(e[1]), np.mean(e[2]), np.mean(e[3]))
        print "-"*n
    return (results, grids)
