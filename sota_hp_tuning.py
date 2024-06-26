from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# 'flare', 'haberman', 'spect', 'ionosphere', 'spectf', 'hungarian', 'diabetes', 'hepatitis',
#                        'appendicitis', 'analcatdata_lawsuit'


results = {}

# parameters = {
#     'max_depth': range (2, 10, 1),
#     'n_estimators': range(60, 220, 40),
#     'learning_rate': [0.1, 0.01, 0.05]
# }

parameters = {'gamma': [0, 0.1, 0.4, 1.6, 3.2,  12.8,  51.2, 200],
                  'learning_rate': [0.01, 0.06, 0.1,  0.2,  0.3, 0.4,  0.6, 0.7],
                  'max_depth': [5,  7,  10,  13, 15],
                  'n_estimators': [50, 80,  115,  150],
                  'reg_alpha': [0, 0.1, 0.4, 1.6, 3.2,  12.8,  51.2, 200],
                  'reg_lambda': [0, 0.1, 0.4, 1.6, 3.2,  12.8,  51.2, 200]
              }


for data_name in tqdm(['flare', 'haberman', 'spect', 'ionosphere', 'spectf', 'hungarian', 'diabetes', 'hepatitis',
                       'appendicitis', 'analcatdata_lawsuit']):

    if data_name == 'pima-indians-diabetes':
        data = pd.read_csv('../data/pima-indians-diabetes.txt')
    else:
        data = pd.read_csv(f"../data/{data_name}.tsv",
                           sep='\t')
    X = data.values[:, :-1]
    y = data.values[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,  stratify=y)

    estimator = XGBClassifier(
        # objective='binary:logistic',
        nthread=4,
        seed=42
    )

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        scoring='f1_weighted',
        n_jobs=10,
        cv=3,
        verbose=True
    )


    grid_search.fit(X_train, y_train)

    print(data_name)
    print(grid_search.best_params_)

    results[data_name] = grid_search.best_params_

print(results)
