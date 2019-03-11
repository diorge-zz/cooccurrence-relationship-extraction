import os
import pandas as pd
from collections import defaultdict
from operator import itemgetter


class InstanceFrequencyCount:
    """Returns the mean of the frequency
    of the instances of the two categories.
    In other words, how common are the
    instances of the categories, on average.

    Uses the raw SVO because it doesn't
    require both S and O to be each of
    one category.
    """
    def __init__(self, cache=False):
        self.cache = cache

    def __repr__(self):
        return 'Instance_frequency_count'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return ['raw_svo']

    def required_data(self):
        return ['cat1', 'cat2']

    def creates(self):
        return ['instance_frequency_cat1', 'instance_frequency_cat2']

    def returns(self):
        return ['mean_instance_frequency_cat1', 'mean_instance_frequency_cat2']

    def apply(self, raw_svo, cat1, cat2, output_dir, **kwargs):
        frequencies1 = self.count(raw_svo, cat1)
        frequencies2 = self.count(raw_svo, cat2)
        
        freq1_df = pd.DataFrame({'instance': list(frequencies1.keys()),
                                 'frequency': list(frequencies1.values())})
        freq2_df = pd.DataFrame({'instance': list(frequencies2.keys()),
                                 'frequency': list(frequencies2.values())})

        # "normalizes" by settings the range 0-1 linearly,
        # ie. divides by the maximum
        freq1_df['normalized'] = freq1_df['frequency'] / freq1_df['frequency'].max()
        freq2_df['normalized'] = freq2_df['frequency'] / freq2_df['frequency'].max()

        # save intermediate step for later inspection
        freq1_df.to_csv(os.path.join(output_dir, 'instance_frequency_cat1'), index=False)
        freq2_df.to_csv(os.path.join(output_dir, 'instance_frequency_cat2'), index=False)

        # actual feature
        mean1 = freq1_df['normalized'].mean()
        mean2 = freq2_df['normalized'].mean()

        return {'mean_instance_frequency_cat1': mean1,
                'mean_instance_frequency_cat2': mean2}

    def count(self, svo, instances):
        counter = defaultdict(lambda: 0)

        with open(svo) as svo_contents:
            for line in svo_contents:
                s, v, o, n = line.split('\t')

                if s in instances:
                    counter[s] += int(n)
                if o in instances:
                    counter[o] += int(n)

        return counter


class Specifity:
    """Feature calculating how specific the relation
    is to the category pair in question
    """
    def __init__(self, cache=False):
        self.cache = cache

    def __repr__(self):
        return 'Specifity'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return ['raw_svo']

    def required_data(self):
        return ['cat1', 'cat2', 'relation_names']

    def creates(self):
        return []

    def returns(self):
        return ['relation_specifity_df']

    def apply(self, raw_svo, cat1, cat2, relation_names, **kwargs):
        counter = {}
        for relation in relation_names:
            counter[relation] = {'cat1_unspecific': 0,
                                 'cat2_unspecific': 0,
                                 'cooccurrence_count': 0,
                                 'cooccurrence_count_question': 0}

        with open(raw_svo) as svo_contents:
            for line in svo_contents:
                s, v, o, n = line.split('\t')
                for relation in relation_names:
                    if v == relation:
                        if s in cat1:
                            if o in cat2:
                                counter[v]['cooccurrence_count'] += 1
                            else:
                                counter[v]['cat1_unspecific'] += 1
                        elif o in cat1:
                            if s in cat2:
                                counter[v]['cat2_unspecific'] += 1
                            else:
                                counter[v]['cooccurrence_count_question'] += 1

        return {'relation_specifity_df': pd.DataFrame(counter).T}


class PatternContextSize:
    def __init__(self, cache=False):
        self.cache = cache

    def __repr__(self):
        return 'Pattern_context_size'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return []

    def required_data(self):
        return ['relation_names', 'groups']

    def creates(self):
        return []

    def returns(self):
        return ['pattern_context_size_df']

    def apply(self, relation_names, groups, **kwargs):
        cluster_sizes = pd.value_counts(pd.Series(groups)).sort_index().values
        pattern_context = pd.DataFrame({'relation': relation_names,
                                        'Pattern Context Size': cluster_sizes})
        pattern_context.set_index('relation', inplace=True)

        return {'pattern_context_size_df': pattern_context}


class RelationshipCharacteristics:
    def __init__(self, cache=False):
        self.cache = cache

    def __repr__(self):
        return 'Relationship_characteristics'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return ['instance_frequency_cat1', 'instance_frequency_cat2']

    def required_data(self):
        return ['group_pairs', 'cat1', 'cat2', 'relation_names']

    def creates(self):
        return []

    def returns(self):
        return ['commonest_instances_frequencies']

    def apply(self, group_pairs, cat1, cat2, relation_names,
              instance_frequency_cat1, instance_frequency_cat2, **kwargs):
        
        final_frequencies = []

        for pairs in group_pairs:
            commonest_c1_inst, cc1inst_count = self.most_cooccurring_category_instance(pairs, cat1)
            commonest_c2_inst, cc2inst_count = self.most_cooccurring_category_instance(pairs, cat2)

            # yes, it is divided by the length of the other category
            normalized_cc1instcount = cc1inst_count / len(cat2)
            normalized_cc2instcount = cc2inst_count / len(cat1)

            c1_frequencies_df = pd.read_csv(instance_frequency_cat1).set_index('instance')
            c2_frequencies_df = pd.read_csv(instance_frequency_cat2).set_index('instance')

            commonest_c1_freq = c1_frequencies_df.loc[commonest_c1_inst, 'frequency']
            commonest_c2_freq = c2_frequencies_df.loc[commonest_c2_inst, 'frequency']

            frequencies = (cc1inst_count, normalized_cc1instcount,
                           cc2inst_count, normalized_cc2instcount)
            final_frequencies.append(frequencies)

        column_names = ['Commonest Cat1 Instance frequency',
                        'Commonest Cat1 Instance normalized frequency',
                        'Commonest Cat2 Instance frequency',
                        'Commonest Cat2 Instance normalized frequency']

        frequencies_df = pd.DataFrame(final_frequencies, columns=column_names,
                                      index=relation_names)
        return {'commonest_instances_frequencies': frequencies_df}

    def most_cooccurring_category_instance(self, pairs, instances):
        cooccurrences = defaultdict(lambda: 0)
        for a, b in pairs:
            # simply sorting which is the instance
            # and the co-occurrence
            if a not in instances:
                a, b = b, a
            instance, cooccurrence = a, b

            cooccurrences[instance] += 1

        return max(cooccurrences.items(), key=itemgetter(1))


class FeatureAggregator:
    def __init__(self, *, save_output=False, cache=False):
        self.save_output = save_output
        self.cache = cache

    def __repr__(self):
        return 'Feature_aggregator'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return []

    def required_data(self):
        return ['relation_names']

    def creates(self):
        if self.save_output:
            return ['classifier_data']
        return []

    def returns(self):
        return ['classification_data']

    def apply(self, relation_names, output_dir, **kwargs):
        current = pd.DataFrame(index=relation_names)

        df_names = ['pattern_context_size_df',
                    'commonest_instances_frequencies',
                    'pattern_specifity_df']
        for df_name in df_names:
            if df_name in kwargs:
                current = current.join(kwargs[df_name])

        if self.save_output:
            current.to_csv(os.path.join(output_dir, 'classifier_data'))
        
        return {'classification_data': current}
