import datetime
import logging
import os

import classifier_features as classifier

import experiment

import ontext

import preproc


CACHE_DIR = os.path.expanduser('~/data/ontext_experiments/cache')
BASE_SVO = os.path.expanduser('~/data/mall/v+prep_svo-triples.txt')
CATEGORY_DIR = os.path.expanduser('~/data/mall/instances')
OUTPUT_BASE_DIR = os.path.expanduser('~/data/ontext_experiments')
DATETIME_FORMAT = '%Y_%m_%d.%H_%M_%S'
LOGGING_FORMAT = '%(levelname)s %(asctime)s %(funcName)s\t%(message)s'


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


def run(category_pairs, output_dir):

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
                     ontext.BuildCooccurrenceMatrix(),
                     ontext.NormalizeMatrix(),
                     ontext.OntextKmeans(k=5),
                     ontext.InstanceRanker(),
                     ontext.EvidenceForPromotion(promoted_instances=50),
                     classifier.InstanceFrequencyCount(),
                     classifier.Specifity(),
                     classifier.PatternContextSize(),
                     classifier.RelationshipCharacteristics(),
                     classifier.FeatureAggregator(save_output=True),
                     SaveMemoryToDisk(['promoted_pairs',
                                       'evidence_sentences']))

            exp = experiment.Experiment(pair_output_dir,
                                        CACHE_DIR,
                                        steps=steps,
                                        prefix='vpreptriples')

            exp.add_file('raw_svo', BASE_SVO)
            exp.add_file('svo', BASE_SVO)
            exp.prepare()
            exp.execute_all()
        except Exception as e:
            print(f'Category pair {cat1}, {cat2} failed')
            print(str(e))


def main():
    now = datetime.datetime.now().strftime(DATETIME_FORMAT)
    output_dir = os.path.join(OUTPUT_BASE_DIR, now)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logging_file = os.path.join(output_dir, 'log')

    logging.basicConfig(filename=logging_file,
                        level=logging.DEBUG,
                        format=LOGGING_FORMAT)

    category_pairs = [('landscapefeatures', 'aquarium'),
                      ('politicianus', 'religion'),
                      ('stateorprovince', 'awardtrophytournament'),
                      ('televisionshow', 'vehicle'),
                      ('hallwayitem', 'sportsleague'),
                      ('geometricshape', 'building'),
                      ('arthropod', 'vertebrate'),
                      ('weatherphenomenon', 'chemical'),
                      ('furniture', 'flooritem'),
                      ('shoppingmall', 'restaurant')]
    run(category_pairs, output_dir)


if __name__ == '__main__':
    main()
