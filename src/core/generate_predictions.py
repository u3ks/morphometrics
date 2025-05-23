import glob

import geopandas as gpd
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from libpysal.graph import read_parquet
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from core.utils import used_keys

from palettable.colorbrewer.qualitative import Set3_12
from sklearn.metrics import davies_bouldin_score, f1_score

from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn import model_selection
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer


def get_level_cut(mapping_level, v = 'v10'):
    
    cluster_mapping = pd.read_parquet(f'/data/uscuni-ulce/processed_data/clusters/cluster_mapping_{v}.pq')

    return cluster_mapping[mapping_level]


def read_train_test(coredir, train_test_iteration, mapping_level, sample_size):
    
    X_train = pd.read_parquet(f'{coredir}processed_data/train_test_data/training_data{train_test_iteration}.pq')
    y = pd.read_parquet(f'{coredir}processed_data/train_test_data/training_labels{train_test_iteration}.pq')
    level_cut = get_level_cut(mapping_level)

    y['final_without_noise'] = y['final_without_noise'].map(level_cut.to_dict())

    ### undersample
    if sample_size > y['final_without_noise'].value_counts().iloc[-1]:
        sample_size = y['final_without_noise'].value_counts().iloc[-1] - 1_000
    
    np.random.seed(123)
    train_indices = []
    classes = y.final_without_noise.unique()
    has_building = ~y.index.str.split('_').str[-1].str.startswith('-')
    
    for cluster in classes:
        random_indices = np.random.choice(np.where((y.final_without_noise == cluster) & (has_building))[0], sample_size, replace=False, )
        train_indices.append(random_indices)
    
    train_indices = np.concat(train_indices)
    X_train = X_train.iloc[train_indices]
    y = y.iloc[train_indices]
    assert y.final_without_noise.isna().sum() == 0

    X_resampled, y_resampled = X_train, y.final_without_noise

    print(y_resampled.value_counts())

    return X_resampled, y_resampled

    
def get_cluster_names(mapping_level):

    if mapping_level == 1:
        cluster_names = {
     1: 'Incoherent Fabric',
     2: 'Coherent Fabric',
}

    elif mapping_level == 2:
        cluster_names ={
    1: 'Incoherent Large-Scale Fabric',
    2: 'Incoherent Small-Scale Fabric',
    3: 'Coherent Interconnected Fabric',
    4: 'Coherent Dense Fabric'
    
}

    elif mapping_level == 3:
        cluster_names = {
    1: "Incoherent Large-Scale Homogeneous Fabric",
    2: "Incoherent Large-Scale Heterogeneous Fabric",
    3: "Incoherent Small-Scale Linear Fabric",
    4: "Incoherent Small-Scale Sparse Fabric",
    5: "Incoherent Small-Scale Compact Fabric",
    6: "Coherent Interconnected Fabric",
    7: "Coherent Dense Disjoint Fabric",
    8: "Coherent Dense Adjacent Fabric"
}
    else:
        raise Exception(f'Clusters at level {mapping_level} not named')

    return cluster_names


def score_predictions(train_test_iteration, mapping_level, model):
    
    level_cut = get_level_cut(mapping_level)
    X_test = pd.read_parquet(f'{coredir}processed_data/train_test_data/testing_data{train_test_iteration}.pq')
    y_test = pd.read_parquet(f'{coredir}processed_data/train_test_data/testing_labels{train_test_iteration}.pq')
    y_test['final_without_noise'] = y_test['final_without_noise'].map(level_cut.to_dict())

    cluster_names = get_cluster_names(mapping_level)
    
    assert y_test.final_without_noise.isna().sum() == 0
    assert (X_test.index == y_test.index).all()
    
    print(y_test.final_without_noise.map(cluster_names).value_counts())

    ## predictions
    predictions = model.predict(X_test)

    weighted = f1_score(y_test, predictions, average='weighted')
    micro = f1_score(y_test, predictions, average='micro')
    macro = f1_score(y_test, predictions, average='macro')
    overall_acc = pd.Series([weighted, micro, macro], index=['Weighted F1', 'Micro F1', 'Macro F1'])

    f1s_vals = f1_score(y_test, predictions, average=None)
    f1s = pd.Series(
        f1s_vals,
        index = [cluster_names[k] for k in sorted(np.unique(predictions))]
    )
    f1s = f1s.sort_values()

    print(overall_acc)
    print(f1s)
    overall_acc.to_csv(f'{coredir}processed_data/results/overall_acc_{mapping_level}_{train_test_iteration}.csv')
    f1s.to_csv(f'{coredir}processed_data/results/class_f1s_{mapping_level}_{train_test_iteration}.csv')
    

def train_model(core_dir, train_test_iteration, mapping_level, sample_size):

    X_resampled, y_resampled = read_train_test(core_dir, train_test_iteration, mapping_level, sample_size)
    
    from sklearn.ensemble import HistGradientBoostingClassifier

    if 'source' in X_resampled.columns:
        categorical_features = ['source']
    else:
        categorical_features = None
    
    model = HistGradientBoostingClassifier(random_state=123, verbose=1,
                                                learning_rate = 0.03,
                                                categorical_features=categorical_features,
                                                # max_depth = None, 
                                                max_iter = 120, 
                                                # max_leaf_nodes=None,
                                                # max_features=.5
                                               )
    model.fit(X_resampled, y_resampled)
    print(model.score(X_resampled, y_resampled))

    
    score_predictions(train_test_iteration, mapping_level, model)


if __name__ == '__main__':

    mapping_level = 3
    sample_size = 600_000
    core_dir = '/data/uscuni-eurofab-overture/'

    for train_test_iteration in range(1, 8):
        train_model(core_dir, train_test_iteration, mapping_level, sample_size)

