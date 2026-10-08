import time
import queue
import threading
from concurrent.futures import Future, wait, FIRST_COMPLETED
from copy import deepcopy
from datetime import datetime

from edca.fitness import individual_fitness
from edca.ea import generate_population, calculate_average_fitness

from edca.evolutionary_algorithm import (
    EvolutionarySearch,
    TimeBudgetExceeded,
    WORST_FITNESS,
    VERBOSE_SAVE_ALL,
    sort_population,
    get_individual_params,
    sort_individuals_configs
)


class _DaemonThreadPool:
    """
    Thread pool whose workers are daemon threads.

    Unlike concurrent.futures.ThreadPoolExecutor, the interpreter does not wait
    for these threads at exit, so a hung or abandoned pipeline can never keep the
    process alive after the search has finished.
    """

    def __init__(self, max_workers):
        self._queue = queue.Queue()
        self._stopped = False
        self._threads = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(max_workers)
        ]
        for t in self._threads:
            t.start()

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            future, fn, args, kwargs = item
            if not future.set_running_or_notify_cancel():
                continue  # cancelled before starting
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)

    def submit(self, fn, *args, **kwargs):
        future = Future()
        self._queue.put((future, fn, args, kwargs))
        return future

    def shutdown(self):
        """ Cancels queued work and tells idle workers to stop. Never blocks. """
        self._stopped = True
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                item[0].cancel()
        for _ in self._threads:
            self._queue.put(None)


class RandomSearch(EvolutionarySearch):
    """
    Random search over the same search space as EvolutionarySearch.

    Each "iteration" samples `population_size` random individuals
    (a batch), evaluates them (in parallel if n_jobs > 1, reusing the
    evaluated-individuals cache) and keeps track of the best pipeline found so
    far. 

    Parameters are the same as EvolutionarySearch, except that the genetic
    operators / parameters are not needed:
        crossover_operator, mutation_operator, prob_mutation,
        prob_mutation_model, prob_crossover, tournament_size, elitism, patience
    `population_size` is the number of random individuals sampled per iteration.
    """

    def __init__(self,
                 config_models,
                 pipeline_config,
                 sampling_generator,
                 X_train,
                 y_train,
                 X_val,
                 y_val,
                 fitness_metric,
                 population_size=10,
                 num_iterations=100,
                 time_budget=-1,
                 filepath='',
                 n_jobs=5,
                 early_stop=None,
                 verbose=-1,
                 seed=42,
                 individual_timeout=None):
        super().__init__(
            config_models=config_models,
            pipeline_config=pipeline_config,
            crossover_operator=None,
            mutation_operator=None,
            sampling_generator=sampling_generator,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            fitness_metric=fitness_metric,
            population_size=population_size,
            elitism=0,
            num_iterations=num_iterations,
            time_budget=time_budget,
            filepath=filepath,
            n_jobs=n_jobs,
            patience=None,
            early_stop=early_stop,
            verbose=verbose,
            seed=seed,
        )
        # Max seconds a single pipeline may run before being abandoned (scored with
        # the worst fitness). If None (default) it is set automatically:
        #   - time_budget defined: the time still available when the pipeline starts
        #   - otherwise (iterations as stop criterion): DEFAULT_ITERATION_TIMEOUT
        self.individual_timeout = individual_timeout
        self._start_times = {}
        self._time_limits = {}

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _sample_batch(self):
        """ Samples a batch of new random individuals """
        batch = generate_population(
            pop_size=self.population_size,
            config=self.config_models,
            pipeline_config=self.pipeline_config,
            sampling_generator=self.sampling_generator,
        ) 
        return batch

    def _update_best(self, evaluated):
        """ Updates the best individual with an evaluated batch. Returns True if improved """
        if not evaluated:
            return False
        evaluated = sort_population(list(evaluated))
        candidate_fit = evaluated[0][1]['fitness']
        if self.best_fit is None or candidate_fit < self.best_fit:
            self.best_fit = candidate_fit
            self.best_individual = deepcopy(evaluated[0][0])
            self.best_fitness_params = deepcopy(evaluated[0][1])
            return True
        return False

    def _save_info(self):
        """ Saves the information about the iteration (best so far + batch average) """
        if 'average_fitness' not in self.bests_info:
            self.bests_info['average_fitness'] = []
            self.bests_info['config'] = []
            for key in self.best_fitness_params.keys():
                self.bests_info[key] = []
        self.bests_info['average_fitness'].append(calculate_average_fitness(self.population))
        self.bests_info['config'].append(self.best_individual.copy())
        for key, value in self.best_fitness_params.items():
            self.bests_info[key].append(value)

    def _log_iteration(self):
        b = self.best_fitness_params
        self._logger.info(
            f"Iteration {self.iteration} >>> Fitness: {b['fitness']:.3f} - "
            f"Data%: {b['train_percentage']:.3f} - Metric: {b['search_metric']:.3f} - "
            f"CPU Time: {b['time_cpu']:.3f} - S%: {b['samples_percentage']:.3f} - "
            f"F%: {b['features_percentage']:.3f} - CDD: {b['balance_metric']:.3f}")

    _POLL_SECONDS = 1.0
    DEFAULT_ITERATION_TIMEOUT = 5*60  # seconds, used when there is no time budget

    def _current_time_limit(self):
        """ Seconds a pipeline starting now is allowed to run """
        if self.individual_timeout is not None:
            return self.individual_timeout
        if self.time_budget != -1:
            elapsed = (datetime.now() - self._start_datetime).total_seconds()
            return max(self.time_budget - elapsed, 0)
        return self.DEFAULT_ITERATION_TIMEOUT

    def _timed_fitness(self, key, individual_config, individual_id):
        """ Runs the fitness in a worker thread, recording when it started """
        self._start_times[key] = time.time()
        self._time_limits[key] = self._current_time_limit()
        return individual_fitness(
            X_train=self.X_train,
            X_val=self.X_val,
            y_train=self.y_train,
            y_val=self.y_val,
            metric=self.fitness_metric,
            pipeline_config=self.pipeline_config,
            individual=individual_config,
            individual_id=individual_id)

    def _failed_params(self, individual_id):
        """ Fitness params for pipelines that crashed or timed out.
        Adjust to match what individual_fitness returns on error. """
        return {
            'individual_id': individual_id,
            'fitness': WORST_FITNESS,
            'search_metric': WORST_FITNESS,
            'train_percentage': WORST_FITNESS,
            'time_cpu': WORST_FITNESS,
            'samples_percentage': WORST_FITNESS,
            'features_percentage': WORST_FITNESS,
            'balance_metric': WORST_FITNESS,
            'proportion_per_class': None,
        }

    def _evaluate_population_parallel(self, population):
        """ Parallel evaluation that never blocks indefinitely on a worker """
        population = sort_individuals_configs(population)
        self.population_evaluated = []
        pending = {}  # str(individual) -> {'future', 'individual', 'count', 'id'}
        self._start_times = {}
        self._time_limits = {}

        def finish(key, params):
            info = pending.pop(key)
            for _ in range(info['count']):
                self.population_evaluated.append((info['individual'], params))
            self._add_individual_to_evaluated(key, params)

        executor = _DaemonThreadPool(self.n_jobs)
        try:
            # submit new individuals / reuse cached ones
            for individual_config, _ in population:
                key = str(individual_config)
                cached = self._individuals_fitness_df.loc[
                    self._individuals_fitness_df.config == key]
                if len(cached) > 0:
                    self.population_evaluated.append(
                        (individual_config.copy(), get_individual_params(cached).copy()))
                elif key in pending:
                    pending[key]['count'] += 1
                else:
                    self.number_evaluated_individuals += 1
                    ind_id = self.number_evaluated_individuals
                    pending[key] = {
                        'future': executor.submit(
                            self._timed_fitness, key, individual_config, ind_id),
                        'individual': individual_config,
                        'count': 1,
                        'id': ind_id,
                    }
                self._check_time_limit()

            # collect results, polling so time limits are always checked
            while pending:
                self._check_time_limit()
                wait([p['future'] for p in pending.values()],
                     timeout=self._POLL_SECONDS, return_when=FIRST_COMPLETED)
                now = time.time()
                for key in list(pending):
                    info = pending[key]
                    fut = info['future']
                    if fut.done():
                        if fut.exception() is not None:
                            self._logger.info(
                                f"Individual {info['id']} raised: {fut.exception()!r}")
                            finish(key, self._failed_params(info['id']))
                        else:
                            finish(key, fut.result())
                    elif (key in self._start_times and key in self._time_limits
                          and now - self._start_times[key] > self._time_limits[key]):
                        self._logger.info(
                            f"Individual {info['id']} exceeded {self._time_limits[key]:.0f}s - abandoned")
                        fut.cancel()
                        finish(key, self._failed_params(info['id']))

        except TimeBudgetExceeded:
            # keep what already finished, don't wait for the rest
            for key in list(pending):
                fut = pending[key]['future']
                if fut.done() and not fut.cancelled() and fut.exception() is None:
                    finish(key, fut.result())
            raise
        finally:
            executor.shutdown()

        return sort_population(self.population_evaluated)

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def random_search(self):
        self.counter_repeated = 0
        self.counter_no_improvement = 0
        self._logger.info('Random Search')

        # start time counter
        self._start_datetime = datetime.now()
        self.pipeline_config['start_datetime'] = time.time()

        # first batch
        self._logger.info('Create and Evaluate Initial Batch')
        try:
            self.population = deepcopy(self._evaluate_population(self._sample_batch()))
            self._update_best(self.population)
        except KeyboardInterrupt:
            raise KeyboardInterrupt('Ctrl-C pressed')
        except TimeBudgetExceeded:
            if not self.population_evaluated:
                raise TimeBudgetExceeded('No time to evaluate the initial batch')
            self.population = deepcopy(sort_population(self.population_evaluated))
            self._update_best(self.population)
            self._save_info()
            self._save_population('Initial_Incomplete')
            self._logger.info(
                f'Time ended. Only {len(self.population)} were evaluated from the initial batch')
            return
        self._save_info()
        self._save_population('Initial')

        try:
            self._logger.info('Start Search for the best pipeline')
            while ((self.time_budget != -1 or self.iteration < self.num_iterations - 1)
                   and self.counter_no_improvement != self.early_stop):

                self._check_time_limit()

                # sample and evaluate a new, independent batch
                batch = self._sample_batch()
                self._check_time_limit()
                try:
                    self.population = self._evaluate_population(batch)
                except TimeBudgetExceeded:
                    # keep whatever finished before running out of time
                    if self.population_evaluated:
                        self.population = sort_population(self.population_evaluated)
                        self._update_best(self.population)
                    raise

                # track the global best
                if self._update_best(self.population):
                    self.counter_no_improvement = 0
                else:
                    self.counter_no_improvement += 1

                self.iteration += 1
                self._log_iteration()
                self._save_info()

                if self.verbose == VERBOSE_SAVE_ALL or (
                        self.verbose != 0 and self.iteration % self.verbose == 0):
                    self._save_population(self.iteration)

                self._check_time_limit()
        except KeyboardInterrupt:
            raise KeyboardInterrupt('Ctrl-C pressed')
        except TimeBudgetExceeded:
            pass

        # save last population if not already saved
        if self.verbose == 0 or (self.verbose != VERBOSE_SAVE_ALL and self.iteration % self.verbose != 0):
            self._save_population(self.iteration)

        self._logger.info(
            f'Search Ended after {self.iteration} iterations with a time of '
            f'{(datetime.now() - self._start_datetime).total_seconds()} seconds')

        self.save_evaluated_individuals()
        if self.best_fit == WORST_FITNESS:
            self.best_individual = None
            self.best_fitness_params = None
            self._logger.info('No pipeline was found. All pipelines resulted in error')
