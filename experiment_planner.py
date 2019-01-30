import experiment
import preproc
import ontext


def main():
    OUTPUT_DIR = '~/data/ontext_experiments/30jan19'
    CACHE_DIR = '~/data/ontext_experiments/cache'
    BASE_SVO = '~/data/mall/sample4m.svo'
    CATEGORY1 = '~/data/mall/instances/politicianus'
    CATEGORY2 = '~/data/mall/instances/religion'

    exp = experiment.Experiment(OUTPUT_DIR, CACHE_DIR,
            experiment.ReadSvo(BASE_SVO),
            experiment.ReadCategory(CATEGORY1, 1),
            experiment.ReadCategory(CATEGORY2, 2),
            preproc.FilterSentencesByOccurrence(5),
            preproc.FilterInstanceInCategory(),
            experiment.SvoMemoryData(),
            ontext.BuildCooccurrenceMatrix(),
            ontext.NormalizeMatrix())

    exp.prepare()
    exp.execute_all()

    print(exp.data['comatrix'])


if __name__ == '__main__':
    main()
