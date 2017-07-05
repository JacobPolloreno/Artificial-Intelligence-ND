# import re
import warnings

from asl_data import SinglesData
from collections import OrderedDict
from typing import List, Dict, Tuple

def recognize(
        models: dict, test_set: SinglesData) -> Tuple[List[Dict[str, float]], List[str]]:
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities: List[Dict[str, float]] = []
    guesses: List[str] = []

    test_sequences: List[Tuple[List, List[int]]] = list(
        test_set.get_all_Xlengths().values())

    for X_test, X_length in test_sequences:
        results: dict = {}
        best_score: float = float('-inf')
        best_guess: str = None

        for word, model in models.items():
            try:
                score = model.score(X_test, X_length)
            except BaseException:
                score = float('-inf')  # word cannot be scored
            results[word] = score

            if score > best_score:
                best_score = score
                best_guess = word

        probabilities.append(results)
        # Check to see if our guess is a variation of a word
        # e.g. 'GO' and 'GO1'
        # if best_guess not in test_set.wordlist:
            # remove digits from word
            # best_guess = re.sub("\d+", "", best_guess)

        guesses.append(best_guess)

    return probabilities, guesses
