"""Module for managing and running experiments
"""


import os
import re


class Experiment:
    """Manages the state of a running experiment
    """
    def __init__(self, output_dir, cache_dir=None, files=None, data=None, *steps):
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self._steps = tuple(steps)
        self._executed_steps = []
        self._pending_execution = list(reversed(self._steps))
        self.files = files or {}
        self.data = data or {}

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
        execution_string = ''
        for step in self._steps:
            path = os.path.join(self.output_dir, str(step))
            os.mkdir(path)
            execution_string += str(step)
            for step_output in step.creates():
                if execution_string + '.' + step_output in cache_filenames:
                    src = os.path.join(os.path.abspath(self.cache_dir),
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
        return '.'.join(str(step) for step in self._executed_steps)

    def execute_step(self, cache):
        """Executes the next step

        :param cache: bool, if should cache results
        """
        if self.steps_pending() == 0:
            raise ValueError('No steps left to execute')

        current_step = self._pending_execution.pop()
        step_output_dir = os.path.join(self.output_dir, str(current_step))

        # checking cache
        if self.cache_dir is not None:
            saved_outputs = len(os.listdir(step_output_dir))
        else:
            saved_outputs = 0
        intended_outputs = len(current_step.creates())

        if intended_outputs > saved_outputs:
            try:
                for required_file in current_step.required_files():
                    if required_file not in self.files:
                        raise ValueError(f'Missing file {required_file} for step {current_step}')
                for required_data in current_step.required_data():
                    if required_data not in self.data:
                        raise ValueError(f'Missing data {required_data} for step {current_step}')

                new_data = current_step.apply(**{**self.files,
                                                 **self.data,
                                                 'output_dir': step_output_dir})
                if new_data is not None:
                    self.data.update(new_data)
            except Exception as e:
                self._pending_execution.append(current_step)
                raise e
        
        self._executed_steps.append(current_step)
        for new_file in current_step.creates():
            new_path = os.path.join(step_output_dir, new_file)
            self.files[new_file] = new_path
            if cache:
                cache_filename = os.path.join(self.cache_dir,
                                              self.executed_string() + '.' + new_file)
                os.symlink(os.path.abspath(new_path), cache_filename)



    def execute_all(self):
        while self.steps_pending() > 0:
            self.execute_step(False)


class FilterSentencesByOccurrence:

    def __init__(self, min_occurrences):
        if min_occurrences <= 0:
            raise ValueError('min_occurrences must be positive')
        self.min_occurrences = min_occurrences


    def __repr__(self):
        return f'Filter_sentences_by_occurrence_{self.min_occurrences}'

    def __str__(self):
        return repr(self)

    @staticmethod
    def parse(representation):
        regex = r'^Filter_sentences_by_occurrence_(\d+)$'
        result = re.findall(regex, representation)
        if len(result) != 1:
            raise ValueError('Invalid representation')

        min_occurrences = int(result[0])
        return FilterSentencesByOccurrence(min_occurrences)

    def required_files(self):
        return ['svo']

    def required_data(self):
        return []

    def creates(self):
        return ['svo']

    def returns(self):
        return []

    def apply(self, output_dir, svo, **kwargs):
        new_svo_path = os.path.join(output_dir, 'svo')
        
        with open(svo, 'r') as old_svo:
            with open(new_svo_path, 'w') as new_svo:
                self._filter(old_svo, new_svo)

    def _filter(self, instream, outstream):
        for line in instream:
            s, v, o, n = line.split('\t')
            if int(n) >= self.min_occurrences:
                outstream.write(line)
