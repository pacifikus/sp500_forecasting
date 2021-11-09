import os.path

import mlflow
import yaml
from gtda.diagrams import PersistenceEntropy
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import TakensEmbedding, SlidingWindow
from sklearn.ensemble import RandomForestRegressor

from utils import save_pickle, load_pickle

config_path = os.path.join('config/config.yaml')
config = yaml.safe_load(open(config_path))['train']


def train_rf_on_tda(X_train, y_train, artefacts_postfix):
    sliding = SlidingWindow(
        size=config['tda']['sliding']['window_size'],
        stride=config['tda']['sliding']['stride']
    )
    X_train, y_train = sliding.fit_transform_resample(X_train, y_train)

    embedder = TakensEmbedding(
        dimension=config['tda']['embedder']['dimension'],
        time_delay=config['tda']['embedder']['delay'],
    )
    X_train = embedder.fit_transform(X_train)

    vr_persistence = VietorisRipsPersistence()
    X_train = vr_persistence.fit_transform(X_train)

    p_entropy = PersistenceEntropy()
    X_train = p_entropy.fit_transform(X_train)

    save_pickle(sliding, f'{config["data"]["sliding_path"]}_{artefacts_postfix}.pickle')
    mlflow.log_artifact(f'{config["data"]["sliding_path"]}_{artefacts_postfix}.pickle')

    save_pickle(embedder, f'{config["data"]["embedder_path"]}_{artefacts_postfix}.pickle')
    mlflow.log_artifact(f'{config["data"]["embedder_path"]}_{artefacts_postfix}.pickle')

    save_pickle(vr_persistence, f'{config["data"]["persistence_path"]}_{artefacts_postfix}.pickle')
    mlflow.log_artifact(f'{config["data"]["persistence_path"]}_{artefacts_postfix}.pickle')

    save_pickle(p_entropy, f'{config["data"]["entropy_path"]}_{artefacts_postfix}.pickle')
    mlflow.log_artifact(f'{config["data"]["entropy_path"]}_{artefacts_postfix}.pickle')

    clf = RandomForestRegressor(random_state=config['seed'])
    clf.fit(X_train, y_train)
    return clf


def transform_data_to_tda(X_test, y_test, artefacts_postfix):
    sliding = load_pickle(f'{config["data"]["sliding_path"]}_{artefacts_postfix}.pickle')
    embedder = load_pickle(f'{config["data"]["embedder_path"]}_{artefacts_postfix}.pickle')
    vr_persistence = load_pickle(f'{config["data"]["persistence_path"]}_{artefacts_postfix}.pickle')
    p_entropy = load_pickle(f'{config["data"]["entropy_path"]}_{artefacts_postfix}.pickle')

    X_test, y_test = sliding.transform_resample(X_test, y_test)
    X_test = embedder.fit_transform(X_test)
    X_test = vr_persistence.fit_transform(X_test)
    X_test = p_entropy.fit_transform(X_test)

    return X_test, y_test
