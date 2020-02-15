import datetime
import logging
import os
from collections import namedtuple
from typing import Dict, List, Tuple

import experiment

import ncm

import numpy as np

import pandas as pd

import preproc


CACHE_DIR = os.path.expanduser('~/data/ontext_experiments/cache')
BASE_SVO = os.path.expanduser('~/data/mall/v+prep_svo-triples.txt')
CATEGORY_DIR = os.path.expanduser('~/data/mall/instances')
OUTPUT_BASE_DIR = os.path.expanduser('~/data/ontext_experiments')
DATETIME_FORMAT = '%Y_%m_%d.%H_%M_%S'
LOGGING_FORMAT = '%(levelname)s %(asctime)s %(funcName)s\t%(message)s'
CATEGORIES_TABLE = os.path.expanduser('~/data/mall/Filt-Relations100')


Relation = namedtuple('Relation', ['cat1', 'cat2', 'name',
                                   'cluster_size', 'examples'])
Context = namedtuple('Context', ['cat1', 'cat2',
                                 'relation', 'context'])


class SaveMemoryToDisk:
    """Writes any data in the pipeline to the disk
    """
    def __init__(self, data_to_save, cache=False):
        self.data_to_save = data_to_save
        self.cache = cache

    def __repr__(self):
        return 'Save_memory_to_disk'

    def required_files(self):
        return []

    def required_data(self):
        return self.data_to_save

    def creates(self):
        return self.data_to_save

    def returns(self):
        return []

    def apply(self, output_dir, **kwargs):
        for data_key, data in kwargs.items():
            if data_key in self.data_to_save:
                filepath = os.path.join(output_dir, data_key)

                with open(filepath, 'w') as output_handle:
                    if issubclass(str, type(data)):
                        output_handle.write(data)
                    else:
                        output_handle.write(str(data))

        return {}


class BuildOutputReports:
    def __init__(self, cache=False):
        self.cache = cache

    def __repr__(self):
        return 'Build_output_reports'

    def required_files(self):
        return []

    def required_data(self):
        return ['relation_names', 'groups',
                'unique_contexts', 'promoted_pairs']

    def creates(self):
        return []

    def returns(self):
        return ['relations_output', 'contexts_output']

    def apply(self,
              cat1_name: str,
              cat2_name: str,
              relation_names: 'np.array[str]',
              groups: 'np.array[int]',
              unique_contexts: 'np.array[str]',
              promoted_pairs: List[List[Tuple[str, str]]],
              **kwargs):
        relations: List[Relation] = []
        contexts: List[Context] = []

        group_indexes, cluster_count = np.unique(groups,
                                                 return_counts=True)

        cluster_sizes: Dict[int, int] = dict(zip(group_indexes,
                                                 cluster_count))

        for group_index, relation_name in enumerate(relation_names):
            if group_index in cluster_sizes:
                relation = Relation(cat1_name,
                                    cat2_name,
                                    relation_name,
                                    cluster_sizes[group_index],
                                    promoted_pairs[group_index])
                relations.append(relation)

        for relation_index, context_name in zip(groups, unique_contexts):
            if relation_index != -1:
                context = Context(cat1_name,
                                  cat2_name,
                                  relation_names[relation_index],
                                  context_name)
                contexts.append(context)

        return {'relations_output': relations,
                'contexts_output': contexts}


def run(category_pairs, output_dir):
    relations: List[Relation] = []
    contexts: List[Context] = []

    for cat1, cat2 in category_pairs:
        try:
            directory_name = '_'.join([cat1, cat2])
            cat1_dir = os.path.join(CATEGORY_DIR, cat1)
            cat2_dir = os.path.join(CATEGORY_DIR, cat2)
            pair_output_dir = os.path.join(output_dir, directory_name)

            steps = (preproc.FilterSentencesByOccurrence(5),
                     preproc.MinimumContextOccurrence(3),
                     preproc.MinimumPairOccurrence(5),
                     experiment.ReadCategories(cat1_dir, cat2_dir),
                     preproc.FilterInstanceInCategory(),
                     experiment.SvoToMemory(),
                     ncm.BuildCooccurrenceGraph(),
                     ncm.NcmHcsw(),
                     ncm.Medoids(),
                     ncm.PromotePairs(),
                     ncm.Pruner(),
                     BuildOutputReports())

            exp = experiment.Experiment(pair_output_dir,
                                        CACHE_DIR,
                                        steps=steps,
                                        prefix='vpreptriples')

            exp.add_file('raw_svo', BASE_SVO)
            exp.add_file('svo', BASE_SVO)
            exp.data['cat1_name'] = cat1
            exp.data['cat2_name'] = cat2
            exp.prepare()
            exp.execute_all()

            relations.extend(exp.data['relations_output'])
            contexts.extend(exp.data['contexts_output'])
        except Exception as e:
            print(f'Category pair {cat1}, {cat2} failed')
            print(str(e))
            raise

    pd.DataFrame(relations).to_csv(output_dir + '/relations.csv', index=False)
    pd.DataFrame(contexts).to_csv(output_dir + '/contexts.csv', index=False)

    return exp


def main():
    now = datetime.datetime.now().strftime(DATETIME_FORMAT)
    output_dir = os.path.join(OUTPUT_BASE_DIR, now)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logging_file = os.path.join(output_dir, 'log')

    logging.basicConfig(filename=logging_file,
                        level=logging.DEBUG,
                        format=LOGGING_FORMAT)

    category_pairs_table = pd.read_table(CATEGORIES_TABLE,
                                         sep='  ',
                                         header=None,
                                         engine='python',
                                         names=['cat1', 'cat2', 'score'],
                                         usecols=['cat1', 'cat2'])
    category_pairs = category_pairs_table.apply(tuple, axis='columns').tolist()
    return run(category_pairs, output_dir)


if __name__ == '__main__':
    main()
