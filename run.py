import datetime
import os
import logging
import experiment
import preproc
import ontext
import classifier_features as classifier


CACHE_DIR = os.path.expanduser('~/data/ontext_experiments/cache')
BASE_SVO = os.path.expanduser('~/data/mall/v+prep_svo-triples.txt')
CATEGORY_DIR = os.path.expanduser('~/data/mall/instances')
OUTPUT_BASE_DIR = os.path.expanduser('~/data/ontext_experiments')
DATETIME_FORMAT = '%Y_%m_%d.%H_%M_%S'
LOGGING_FORMAT='%(levelname)s %(asctime)s %(funcName)s\t%(message)s'

def run(category_pairs, output_dir):

    for cat1, cat2 in category_pairs:
        directory_name = '_'.join([cat1, cat2])
        cat1_dir = os.path.join(CATEGORY_DIR, cat1)
        cat2_dir = os.path.join(CATEGORY_DIR, cat2)
        pair_output_dir = os.path.join(output_dir, directory_name)

        exp = experiment.Experiment(pair_output_dir, CACHE_DIR,
                steps=(
                    preproc.FilterSentencesByOccurrence(5),
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
                    classifier.FeatureAggregator(save_output=True)),
                prefix='vpreptriples')

        exp.add_file('raw_svo', BASE_SVO)
        exp.add_file('svo', BASE_SVO)
        exp.prepare()
        exp.execute_all()


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
