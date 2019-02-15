import datetime
import os
import experiment
import preproc
import ontext
import classifier_features as classifier


def main():
    CACHE_DIR = '~/data/ontext_experiments/cache'
    BASE_SVO = '~/data/mall/sample4m.svo'
    CATEGORY1 = '~/data/mall/instances/politicianus'
    CATEGORY2 = '~/data/mall/instances/religion'
    OUTPUT_BASE_DIR = '~/data/ontext_experiments'
    DATETIME_FORMAT = '%Y_%m_%d.%H_%M_%S'

    now = datetime.datetime.now().strftime(DATETIME_FORMAT)
    output_dir = os.path.join(OUTPUT_BASE_DIR, now)

    exp = experiment.Experiment(output_dir, CACHE_DIR,
            experiment.ReadCategories(CATEGORY1, CATEGORY2),
            preproc.FilterSentencesByOccurrence(5),
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
            )

    exp.add_file('raw_svo', BASE_SVO)
    exp.add_file('svo', BASE_SVO)
    exp.prepare()
    exp.execute_all()

    print(exp.data['relation_names'])
    print(exp.data['instances_scores'])
    print(exp.data['promoted_pairs'])
    print(exp.data['commonest_instances_frequencies'])


if __name__ == '__main__':
    main()
