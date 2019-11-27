import os
import numpy as np
from random import randint
from stable_baselines.common.vec_env import SubprocVecEnv
from scipy.stats import ttest_ind
from statistics import mean, stdev


class Experiment:
    def __init__(self, config1, config2, sample_size=50, path='.'):
        self.configs = config1, config2
        self.seeds = [randint(0, 10e8) for _ in range(sample_size)]
        self.path = path

    def run(self, analyze_curve=False):
        all_results = []

        for i, config in enumerate(self.configs):
            results = []

            for seed in self.seeds:
                _, means, stdevs = config.run(path=self.path + f'/cfg{i}',
                                              seed=seed)

                results.append(means)

            all_results.append(results)

        all_results = np.moveaxis(np.array(all_results), -1, 0)

        if not analyze_curve:
            all_results = all_results[-1:]

        statistics, p_values = [], []

        for evaluation in all_results:
            statistic, p_value = ttest_ind(*evaluation, equal_var=False)

            statistics.append(statistic)
            p_values.append(p_value)

        return all_results, statistics, p_values


class Configuration:
    def __init__(self, env_builder, model_builder,
                 before=None, after=None, each_eval=None,
                 train_steps=30 * 333334, eval_steps=30 * 33334,
                 eval_frequency=30 * 33334, num_processes=1):
        self.env_builder = env_builder
        self.model_builder = model_builder
        self.num_processes = num_processes
        self.train_steps, self.eval_steps = train_steps, eval_steps
        self.eval_frequency = eval_frequency
        self.before, self.after, self.each_eval = before, after, each_eval

    def run(self, path='.', seed=None):
        env = [lambda: self.env_builder(seed)
               for _ in range(self.num_processes)]
        env = SubprocVecEnv(env, start_method='spawn')

        model = self.model_builder(env)

        if self.before is not None:
            self.before(model, env)

        os.makedirs(path + '/' + str(seed), exist_ok=True)
        model.save(path + '/' + str(seed) + '/0-steps')

        means, stdevs = [], []

        def evaluate(model):
            episode_rewards = [[0.0] for _ in range(env.num_envs)]
            num_steps = int(self.eval_steps / self.num_processes)

            obs = env.reset()

            for j in range(num_steps):
                actions, _ = model.predict(obs, deterministic=True)

                obs, rewards, dones, _ = env.step(actions)

                for i in range(env.num_envs):
                    episode_rewards[i][-1] += rewards[i]

                    if dones[i]:
                        episode_rewards[i].append(0.0)

            all_rewards = []

            for part in episode_rewards:
                all_rewards.extend(part)

            return mean(all_rewards), stdev(all_rewards)

        def callback(_locals, _globals):
            model = _locals['self']
            timestep = _locals["timestep"] + 1

            if timestep % self.eval_frequency == 0:
                mean, std = evaluate(model)

                means.append(mean)
                stdevs.append(std)

                if self.each_eval is not None:
                    self.each_eval(model, env)

                model.save(path + '/' + str(seed) + f'/{timestep}-steps')

        model.learn(total_timesteps=self.train_steps,
                    callback=callback, seed=seed)

        mean_reward, std_reward = evaluate(model)

        means.append(mean_reward)
        stdevs.append(std_reward)

        model.save(path + '/' + str(seed) + '/final')

        if self.after is not None:
            self.after(model, env)

        env.close()

        return model, means, stdevs