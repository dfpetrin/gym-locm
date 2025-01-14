from setuptools import setup

setup(name='gym_locm',
      version='1.3.0',
      install_requires=['gym', 'numpy', 'prettytable', 'pexpect', 'sty'],
      extras_require={
            'experiments': ['numpy', 'scipy', 'stable_baselines', 'hyperopt',
                            'mplcursors', 'pandas', 'matplotlib', 'scikit-learn',
                            'stable_baselines3', 'sb3-contrib', 'wandb'],
      },
      entry_points={
            'console_scripts': [
                  'locm-runner=gym_locm.toolbox.runner:run'
            ]
      })
