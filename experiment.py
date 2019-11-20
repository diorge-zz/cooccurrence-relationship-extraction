"""Module for managing and running experiments
"""


import logging
import os
import shutil
from collections import defaultdict

import numpy as np


class Experiment:
    """Manages the state of a running experiment
    """
    def __init__(self, output_dir, cache_dir, steps, prefix=''):
        """The prefix is used to identify the cache step;
        it should be used to guide the cache w.r.t. the base files
        """
        self.output_dir = os.path.expanduser(output_dir)
        if cache_dir is not None:
            self.cache_dir = os.path.expanduser(cache_dir)
        else:
            self.cache_dir = None
        self._steps = tuple(steps)
        self._executed_steps = []
        self._pending_execution = list(reversed(self._steps))
        self.files = {}
        self.data = {}
        self.prefix = prefix

    def add_file(self, name, path):
        self.files[name] = os.path.expanduser(path)

    def prepare(self):
        """Reads the cache and creates directory structure
        """
        if self.cache_dir is not None:
            cache_filenames = set(os.listdir(self.cache_dir))
        else:
            cache_filenames = set()

        # creates a directory for each step.
        # if the output of a step is in the cache,
        # then creates a symbolic link to the cache
        execution_string = self.prefix + '.'
        for step in self._steps:
            path = os.path.join(self.output_dir, str(step))
            if os.path.exists(path):
                logging.warning('Output directory already exists;'
                                'removing current contents')
                shutil.rmtree(path)
            logging.debug(f'Creating directory {path}')
            os.makedirs(path)
            execution_string += str(step)
            for step_output in step.creates():
                cache_file = execution_string + '.' + step_output
                logging.debug(f'Checking for cache file {cache_file}')
                if cache_file in cache_filenames:
                    logging.info(f'Linking cache file {cache_file}')
                    src = os.path.join(os.path.expanduser(self.cache_dir),
                                       execution_string + '.' + step_output)
                    os.symlink(src, os.path.join(path, step_output))
            execution_string += '.'

    def steps_pending(self):
        """Returns the amount of steps pending execution
        """
        return len(self._pending_execution)

    def executed_string(self):
        """Returns the string of the executed steps so far,
        used for caching results
        """
        return self.prefix + '.' + '.'.join(str(step)
                                            for step in self._executed_steps)

    def execute_step(self):
        """Executes the next step
        """
        if self.steps_pending() == 0:
            raise ValueError('No steps left to execute')

        current_step = self._pending_execution.pop()
        cache = current_step.cache
        step_output_dir = os.path.join(self.output_dir, str(current_step))

        logging.info(str(current_step))

        # checking cache
        if cache and self.cache_dir is not None:
            saved_outputs = len(os.listdir(step_output_dir))
        else:
            saved_outputs = 0
        intended_outputs = len(current_step.creates())
        creates_memory_objects = len(current_step.returns()) > 0

        logging.debug((f'Saved outputs {saved_outputs} | '
                       f'Intended outputs {intended_outputs} | '
                       f'Creates mem obj {creates_memory_objects}'))

        if creates_memory_objects or intended_outputs > saved_outputs:
            logging.debug('Executing step')
            for required_file in current_step.required_files():
                if required_file not in self.files:
                    raise ValueError(f'Missing file {required_file}'
                                     f'for step {current_step}')
            for required_data in current_step.required_data():
                if required_data not in self.data:
                    raise ValueError(f'Missing data {required_data}'
                                     f'for step {current_step}')

            args = {**self.files, **self.data, 'output_dir': step_output_dir}
            new_data = current_step.apply(**args)

            if new_data is not None:
                self.data.update(new_data)
        else:
            logging.info('Step skipped, using cache')

        self._executed_steps.append(current_step)
        for new_file in current_step.creates():
            new_path = os.path.join(step_output_dir, new_file)
            self.files[new_file] = new_path
            if cache:
                cache_filename = self.executed_string() + '.' + new_file
                cache_path = os.path.join(self.cache_dir, cache_filename)
                if not os.path.exists(cache_path):
                    os.symlink(os.path.expanduser(new_path), cache_path)

    def execute_all(self):
        while self.steps_pending() > 0:
            self.execute_step()


class ReadCategories:
    def __init__(self, path1, path2, cache=False):
        self.path1 = path1
        self.category1 = os.path.basename(path1)
        self.path2 = path2
        self.category2 = os.path.basename(path2)
        self.cache = cache

    def __repr__(self):
        return f'Read_categories_{self.category1}_{self.category2}'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return []

    def required_data(self):
        return []

    def creates(self):
        return []

    def returns(self):
        return ['cat1', 'cat2']

    def apply(self, **kwargs):
        return {'cat1': set(self.load(self.path1)),
                'cat2': set(self.load(self.path2))}

    def load(self, path):
        with open(os.path.expanduser(path)) as entries:
            for entry in entries:
                yield entry.strip()


class SvoToMemory:
    """After the SVO has been preprocessed,
    load the remaining values into memory indexes
    """
    def __init__(self, cache=False):
        self.cache = cache

    def __repr__(self):
        return 'Svo_to_memory'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return ['svo']

    def required_data(self):
        return []

    def creates(self):
        return []

    def returns(self):
        return ['pair_to_contexts', 'contexts_to_pairs', 'unique_contexts']

    def apply(self, svo, **kwargs):
        pair_to_contexts = defaultdict(lambda: [])
        contexts_to_pairs = defaultdict(lambda: [])
        unique_contexts = set()

        with open(svo) as svo_contents:
            for line in svo_contents:
                s, v, o, n = line.split('\t')
                n = int(n)
                pair = tuple(sorted([s, o]))
                rev = pair == tuple([s, o])
                pair_to_contexts[pair].append((v, n, rev))
                contexts_to_pairs[v].append((pair, n))
                unique_contexts.add(v)

        ucontexts_array = np.array(sorted(unique_contexts))

        return {'pair_to_contexts': pair_to_contexts,
                'contexts_to_pairs': contexts_to_pairs,
                'unique_contexts': ucontexts_array}
