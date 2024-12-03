# NK Model for Human-AI Partnerships

## Purpose
This project extends on the existing NK model to model AI Assistant Agents working with Human Agents as pairs. In a overarching view, it models:

1. AI Assistants (with enhanced search capabilities and several AI-specific parameters)
2. Human Agents and AI Assistant Agents working in pairs to navigate the landscape
3. Cognitive landscapes for both Human and AI Assistant Agents

Please refer to the code for more details.

## Folder 1: Experiment
This folder includes the code for running experiments across different parameters. In each experiment with the given parameter, there will be 100 landscapes * 1000 agents. Agents will have randomized characteristics (as specified in the code). The bits and fitness across time will be stored for the experiment agents and the counterfactual agents.

## Folder 2: Data Processing
This is for processing the output data. There are a few steps:

_data\_generation.py_: converts data from each experiment to long from and add 'k' and 'overlap' data
_data\_generation\_s2.py_: merge data from all experiments into one, and furnish with landscape data as a categorical value.

_At the end of these 2 steps, we will get a long data with each row corresponding to one data instance at a specific time, specific k value, specific overlap value, specific landscape value._

_fitness\_result.py_: Process data and build plots for fitness
_fitness\_result\_lr.py_: Linear regression for fitness
_novelty\_result.py_: Process data and build plots for novelty
_novelty\_result\_lr.py_: Linear regression for novelty

