"""Steps for preprocessing
"""


import logging
import os
from collections import defaultdict


logger = logging.getLogger(__name__)


class FilterSentencesByOccurrence:
    def __init__(self, min_occurrences, cache=True):
        if min_occurrences <= 0:
            raise ValueError('min_occurrences must be positive')
        self.min_occurrences = min_occurrences
        self.cache = cache

    def __repr__(self):
        return f'Filter_sentences_by_occurrence_{self.min_occurrences}'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return ['svo']

    def required_data(self):
        return []

    def creates(self):
        return ['svo']

    def returns(self):
        return []

    def apply(self, output_dir, svo, **kwargs):
        new_svo_path = os.path.join(output_dir, 'svo')

        with open(svo, 'r') as old_svo:
            with open(new_svo_path, 'w') as new_svo:
                self._filter(old_svo, new_svo)

    def _filter(self, instream, outstream):
        for line in instream:
            s, v, o, n = line.split('\t')
            if int(n) >= self.min_occurrences:
                outstream.write(line)


class FilterInstanceInCategory:
    """Filters the SVO to only sentences
    within the two categories
    """
    def __init__(self, reverse=True, cache=True):
        self.reverse = reverse
        self.cache = cache

    def __repr__(self):
        if not self.reverse:
            return 'Filter_instance_in_category'
        return 'Filter_instance_in_category_oneway'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return ['svo']

    def required_data(self):
        return ['cat1', 'cat2']

    def creates(self):
        return ['svo']

    def returns(self):
        return []

    def apply(self, output_dir, svo, cat1, cat2, **kwargs):
        new_svo_path = os.path.join(output_dir, 'svo')
        with open(new_svo_path, 'w') as outstream:
            with open(svo, 'r') as svo_contents:
                for line in svo_contents:
                    s, v, o, n = line.split('\t')
                    lefttoright = s in cat1 and o in cat2
                    righttoleft = self.reverse and o in cat1 and s in cat2

                    if lefttoright or righttoleft:
                        outstream.write(line)


class MinimumContextOccurrence:
    """Filters SVO to only contexts
    that happens in a minimum number of
    different sentences.

    Makes two passes in the SVO
    """
    def __init__(self, minimum_sentences, cache=True):
        if minimum_sentences <= 0:
            raise ValueError('minimum_sentences must be positive')
        self.minimum_sentences = minimum_sentences
        self.cache = cache

    def __repr__(self):
        return f'Minimum_context_occurrence_{self.minimum_sentences}'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return ['svo']

    def required_data(self):
        return []

    def creates(self):
        return ['svo']

    def returns(self):
        return []

    def apply(self, output_dir, svo, **kwargs):
        with open(svo, 'r') as svo_file:
            occ = self.count(svo_file)

        new_svo_path = os.path.join(output_dir, 'svo')

        input_size = 0
        output_size = 0

        with open(new_svo_path, 'w') as outstream:
            with open(svo, 'r') as instream:
                for line in instream:
                    s, v, o, n = line.split('\t')
                    input_size += 1
                    if occ[v] >= self.minimum_sentences:
                        outstream.write(line)
                        output_size += 1

        logger.debug(f'Applied self=<{repr(self)}>'
                     f' filtering input_size=<{input_size}> lines'
                     f' to output_size=<{output_size}> lines')

    def count(self, svo_file):
        occurrences = defaultdict(lambda: 0)
        for line in svo_file:
            s, v, o, n = line.split('\t')
            occurrences[v] += 1

        return occurrences


class MinimumPairOccurrence:
    """Filter sentences with (S, O) pairs
    that do not appear in a minimum of different sentences
    """
    def __init__(self, minimum, cache=True):
        if minimum <= 1:
            raise ValueError('minimum must be at least 2')
        self.minimum = minimum
        self.cache = cache

    def __repr__(self):
        return f'Minimum_pair_occurrence_{self.minimum}'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return ['svo']

    def required_data(self):
        return []

    def creates(self):
        return ['svo']

    def returns(self):
        return []

    def apply(self, output_dir, svo, **kwargs):
        with open(svo) as svo_contents:
            occ = self.count(svo_contents)

        new_svo_path = os.path.join(output_dir, 'svo')
        with open(new_svo_path, 'w') as outstream:
            with open(svo, 'r') as instream:
                for line in instream:
                    s, v, o, n = line.split('\t')
                    pair = frozenset([s, o])
                    if occ[pair] >= self.minimum:
                        outstream.write(line)

    def count(self, svo):
        occurrences = defaultdict(lambda: 0)
        for line in svo:
            s, v, o, n = line.split('\t')
            pair = frozenset([s, o])
            occurrences[pair] += 1
        return occurrences
