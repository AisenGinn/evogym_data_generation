# Large Language Models as Natural Selector for Embodied Soft Robot Design

This project is built upon the [Evogym](https://github.com/evogym/evogym) framework to generate data using genetic algorithms. The generated data is utilized to create inputs for language models (LLMs), and the performance of various LLMs is evaluated through provided scripts.

## About Evogym

Evogym is a simulation environment for evolutionary robotics, enabling the design and testing of soft robots in diverse scenarios. This project leverages Evogym for data generation, forming the foundation of our experiments. For more details about Evogym, visit the [Evogym repository](https://github.com/evogym/evogym). 

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**
```bash
git clone https://github.com/AisenGinn/evogym_data_generation.git
```
2. **Install dependencies**
Please follow the official [Evogym repository](https://github.com/evogym/evogym) to install denpendencies. Additional api depencies are listed here:
```bash
# install OpenAI api dependency
pip install openai

# install Gemini api denpendency
pip install google
```

## Usage
The project includes scripts for data generation and evaluation. Below are instructions to run the key scripts:

- **Data Generation with Genetic Algorithm.**
To generate data using a genetic algorithm, change the data saving path in [ga_data_generation.py](data_generator/ga_data_generation.py) to your desire directory, then run:
```bash
python data_generator/ga_data_generation.py
```

- **Question generation.**
To generate question according to the data, change the question saving path in [llminput_gen.py](data_generator/llminput_gen.py) to your desire directory, then run:
```bash
python data_generator/llminput_gen.py
```

- **Model evaluation.**
To generate answer from LLMs output, change the answer saving path in the files to your desire directory, then run:
```bash
python data_generator/MODEL_NAME_api_eal.py
```

- **Performance evaluation**
To evaluate the performance of LLMs answer, change the answer saving path in the files to your own path, then run:
```bash
# Accuracy evaluation
python data_generator/metrics/compute_acc.py

# Consistency evaluation
python data_generator/metrics/compute_cons.py

# Difficulty Weighted Accuracy evaluation
python data_generator/metrics/compute_dwa.py
```