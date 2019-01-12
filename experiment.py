"""Module for managing and running experiments
"""


import os


class FilterSentencesByOccurrence:

    OUTPUT_SUBDIRECTORY = 'filter_sentences_by_occurrence'
    NEW_SVO_FILENAME = 'filtered.svo'

    def __init__(self, min_occurrences):
        assert min_occurrences > 0
        self.min_occurrences = min_occurrences


    def __repr__(self):
        return f'Filter sentences by occurrence ({self.min_occurrences})'


    def apply(self, state):
        assert 'svo' in state
        assert 'output_directory' in state

        output_dir = os.path.join(state['output_directory'],
                                  self.OUTPUT_SUBDIRECTORY)
        os.makedirs(output_dir)
        svo_path = os.path.join(output_dir, self.NEW_SVO_FILENAME)
        
        # TODO: write to svo_path

        return {'svo': svo_path}


def run_experiment(steps, output_directory):
    current_state = {'output_directory': output_directory}

    for step in steps:
        state_change = step.apply(current_state)
        current_state.update(state_change)

    return current_state
