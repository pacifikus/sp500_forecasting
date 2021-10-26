from gtda.diagrams import PersistenceEntropy
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import TakensEmbedding, SlidingWindow
from sklearn.ensemble import RandomForestRegressor
from utils import save_pickle, load_pickle


def train_rf_on_tda(X_train, y_train, params, artefacts_postfix):
    sliding = SlidingWindow(size=params['window_size'], stride=params['stride'])
    X_train, y_train = sliding.fit_transform_resample(X_train, y_train)

    embedder = TakensEmbedding(
        dimension=params['dimension'],
        time_delay=params['delay']
    )
    X_train = embedder.fit_transform(X_train)

    vr_persistence = VietorisRipsPersistence()
    X_train = vr_persistence.fit_transform(X_train)

    p_entropy = PersistenceEntropy()
    X_train = p_entropy.fit_transform(X_train)

    save_pickle(sliding, f'data/models/sliding/{artefacts_postfix}.pickle')
    save_pickle(embedder, f'data/models/embedder/{artefacts_postfix}.pickle')
    save_pickle(vr_persistence, f'data/models/vr_persistence/{artefacts_postfix}.pickle')
    save_pickle(p_entropy, f'data/models/p_entropy/{artefacts_postfix}.pickle')

    clf = RandomForestRegressor(random_state=params['seed'])
    clf.fit(X_train, y_train)
    return clf


def transform_data_to_tda(X_test, y_test, artefacts_postfix):
    sliding = load_pickle(f'data/models/sliding/{artefacts_postfix}.pickle')
    embedder = load_pickle(f'data/models/embedder/{artefacts_postfix}.pickle')
    vr_persistence = load_pickle(f'data/models/vr_persistence/{artefacts_postfix}.pickle')
    p_entropy = load_pickle(f'data/models/p_entropy/{artefacts_postfix}.pickle')

    X_test, y_test = sliding.transform_resample(X_test, y_test)
    X_test = embedder.fit_transform(X_test)
    X_test = vr_persistence.fit_transform(X_test)
    X_test = p_entropy.fit_transform(X_test)

    return X_test, y_test
