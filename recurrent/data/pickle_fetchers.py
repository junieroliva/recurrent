import cPickle as pickle
import recurrent.experiments.fetchers as fetchers
from recurrent.model.sequence_model import TEST, TRAIN, VALID


def make_pickled_fetchers(data_location, batch_sizes,
                          fetcher_class=fetchers.WholeFetcher, window=None,
                          random_shuffle=True, state_is_tuple=False,
                          shift_sequences=True):
    datasets = pickle.load(open(data_location, 'rb'))
    fetchers = {TRAIN: None, VALID: None, TEST: None}
    for dset in datasets:
        if dset == TEST:
            loop_back = True
        else:
            loop_back = False
        fetchers[dset] = fetcher_class(datasets[dset], batch_sizes[dset],
                                       window=window,
                                       random_shuffle=random_shuffle,
                                       state_is_tuple=state_is_tuple,
                                       loop_back=loop_back,
                                       shift_sequences=shift_sequences)
    return fetchers
