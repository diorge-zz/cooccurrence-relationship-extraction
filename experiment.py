"""Module for managing and running experiments
"""


import os


class Experiment:
    """Manages the state of a running experiment
    """
    def __init__(self, output_dir, cache_dir, *steps):
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
        return '.'.join(str(step) for step in self._executed_steps)

    def execute_step(self):
        """Executes the next step

        :param cache: bool, if should cache results
        """
        if self.steps_pending() == 0:
            raise ValueError('No steps left to execute')

        current_step = self._pending_execution.pop()
        cache = current_step.cache
        step_output_dir = os.path.join(self.output_dir, str(current_step))

        # checking cache
        if cache and self.cache_dir is not None:
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
                os.symlink(os.path.expanduser(new_path), cache_filename)

    def execute_all(self):
        while self.steps_pending() > 0:
            self.execute_step()


class ReadSvo:
    def __init__(self, svopath, cache=False):
        self.svopath = svopath
        self.alias = os.path.basename(svopath)
        self.cache = cache

    def __repr__(self):
        return f'Read_svo_{self.alias}'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return []

    def required_data(self):
        return []

    def creates(self):
        return ['svo']

    def returns(self):
        return []

    def apply(self, output_dir, **kwargs):
        write_path = os.path.join(output_dir, 'svo')
        os.symlink(os.path.expanduser(self.svopath), os.path.expanduser(write_path))


class ReadCategories:
    def __init__(self, path, number=1, cache=False):
        self.path = path
        self.category = os.path.basename(path)
        self.number = number
        self.cache = cache

    def __repr__(self):
        return f'Read_category_{self.category}_{self.number}'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return []

    def required_data(self):
        return []

    def creates(self):
        return []

    def returns(self):
        return [f'cat{self.number}']

    def apply(self, **kwargs):
        return {f'cat{self.number}': tuple(self.load())}

    def load(self):
        with open(os.path.expanduser(self.path)) as entries:
            for entry in entries:
                yield entry.strip()
