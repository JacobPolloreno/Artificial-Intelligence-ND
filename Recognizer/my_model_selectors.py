import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from typing import Tuple, Union
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(
            self,
            all_word_sequences: dict,
            all_word_Xlengths: dict,
            this_word: str,
            n_constant=3,
            min_n_components=2,
            max_n_components=10,
            random_state=14,
            verbose=False) -> None:
        self.words: dict = all_word_sequences
        self.hwords: dict = all_word_Xlengths
        self.sequences: dict = all_word_sequences[this_word]
        # self.X : dict
        self.lengths: int
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word: str = this_word
        self.n_constant: int = n_constant
        self.min_n_components: int = min_n_components
        self.max_n_components: int = max_n_components
        self.random_state: int = random_state
        self.verbose: bool = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states: int) -> Union[GaussianHMM, None]:
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(
                n_components=num_states,
                covariance_type="diag",
                n_iter=1000,
                random_state=self.random_state,
                verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(
                    self.this_word, num_states))
            return hmm_model
        except BaseException:
            if self.verbose:
                print("failure on {} with {} states".format(
                    self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self) -> GaussianHMM:
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components: int = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self) -> GaussianHMM:
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        def bic_score(logL: float, num_params: int,
                      num_data: int) -> float:
            '''Maximizes likelihood of the data while penalizing large-size
            models
                BIC = -2 * logL + p * logN

            :param logL: float, likelihood of fitted param
            :param p: int, number of parameters
            :param num_data: int, number of data points
            '''
            return -2 * logL + num_params * np.log(num_data)

        best_bic_score: float = float('-inf')
        best_model: GaussianHMM = None

        # number of data points to calculate bic score
        num_data: int = len(self.X)
        num_features: int = len(self.X[0])

        # iterate over possible components to find best num
        for num_components in range(self.min_n_components,
                                    self.max_n_components + 1):
            try:
                # train model
                model: GaussianHMM = GaussianHMM(
                    n_components=num_components,
                    covariance_type="diag",
                    n_iter=1000,
                    random_state=self.random_state).fit(
                        self.X,
                        self.lengths)

                # evaluate model
                logL: float = model.score(self.X, self.lengths)

                # calculate number of params
                num_params = num_components**2 + 2 * num_components * num_features - 1

                # compute bic
                score: float = bic_score(logL, num_params, num_data)

                if score > best_bic_score:
                    best_bic_score = score
                    best_model = model
            except BaseException:
                continue

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self) -> GaussianHMM:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        best_dic_score: float = float('-inf')
        best_model: GaussianHMM = None

        # get competing words
        other_words = self.hwords.copy()  # make shallow copy
        del other_words[self.this_word]  # remove current word

        for num_components in range(self.min_n_components,
                                    self.max_n_components + 1):
            other_words_scores: list = []

            # get log likelihoods of competing worlds
            for X_other, lengths_other in other_words.values():
                try:
                    # train competing model
                    model = GaussianHMM(
                        n_components=num_components,
                        covariance_type="diag",
                        n_iter=1000,
                        random_state=self.random_state).fit(
                            X_other,
                            lengths_other)

                    # score model
                    score: float = model.score(X_other, lengths_other)
                    other_words_scores.append(score)
                except BaseException:
                    continue

            # get log likelihood of current world
            try:
                # train current model
                model = GaussianHMM(
                    n_components=num_components,
                    covariance_type="diag",
                    n_iter=1000,
                    random_state=self.random_state).fit(
                        self.X,
                        self.lengths)

                # evaluate model
                score = model.score(self.X, self.lengths)

                # compute dic
                dic_score = score - np.mean(other_words_scores)

                if dic_score > best_dic_score:
                    best_dic_score = dic_score
                    best_model = model
            except BaseException:
                continue

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self) -> Union[None, GaussianHMM]:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # init "best" model and score
        best_num_components: int = 0
        best_avg_score: float = float('-inf')

        # iterate over possible components to find best num
        for num_components in range(self.min_n_components,
                                    self.max_n_components + 1):
            # array of log scores
            scores: list = []

            try:
                split_method = KFold(
                    random_state=self.random_state,
                    n_splits=min(3, len(self.sequences)))

                # split sequences of words into training and testing sets
                for cv_train_idx, cv_test_idx in split_method.split(
                        self.sequences):
                    # create train and test sequences
                    X_train, lengths_train = combine_sequences(cv_train_idx,
                                                               self.sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx,
                                                             self.sequences)

                    try:
                        # train model
                        model = GaussianHMM(
                            n_components=num_components,
                            covariance_type="diag",
                            n_iter=1000,
                            random_state=self.random_state,
                            verbose=False).fit(
                            X_train,
                            lengths_train)

                        # evaluate model
                        score = model.score(X_test, lengths_test)
                        scores.append(score)
                    except BaseException:
                        continue
            except BaseException:
                continue

            if np.mean(scores) > best_avg_score:
                best_num_components = num_components
                best_avg_score = np.mean(scores)

        if best_num_components != 0:
            # return GaussianHMM model trained on all data with best num of
            # components
            return self.base_model(best_num_components)
        else:
            return None
