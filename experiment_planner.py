import datetime
import os
import experiment
import preproc
import ontext


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
            experiment.ReadCategory(CATEGORY1, 1),
            experiment.ReadCategory(CATEGORY2, 2),
            preproc.FilterSentencesByOccurrence(5),
            preproc.FilterInstanceInCategory(),
            experiment.SvoToMemory(),
            ontext.BuildCooccurrenceMatrix(),
            ontext.NormalizeMatrix(),
            ontext.OntextKmeans(k=5),
            ontext.InstanceRanker())

    exp.add_file('svo', BASE_SVO)
    exp.prepare()
    exp.execute_all()

    print(exp.data['relation_names'])
    print(exp.data['instances_scores'])


if __name__ == '__main__':
    main()
