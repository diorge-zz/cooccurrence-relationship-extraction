"""Steps for preprocessing
"""


import os


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
        self.reverse = True
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
