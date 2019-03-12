import datetime
import os
import experiment
import preproc
import ontext
import classifier_features as classifier


CACHE_DIR = '~/data/ontext_experiments/cache'
BASE_SVO = '~/data/mall/v+prep_svo-triples.txt'
CATEGORY_DIR = '~/data/mall/instances'
OUTPUT_BASE_DIR = '~/data/ontext_experiments'
DATETIME_FORMAT = '%Y_%m_%d.%H_%M_%S'

def run(category_pairs):
    now = datetime.datetime.now().strftime(DATETIME_FORMAT)

    for cat1, cat2 in category_pairs:
        directory_name = '_'.join([cat1, cat2])
        cat1_dir = os.path.join(CATEGORY_DIR, cat1)
        cat2_dir = os.path.join(CATEGORY_DIR, cat2)
        output_dir = os.path.join(OUTPUT_BASE_DIR, now, directory_name)

        exp = experiment.Experiment(output_dir, CACHE_DIR,
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
    run(category_pairs)


if __name__ == '__main__':
    main()
