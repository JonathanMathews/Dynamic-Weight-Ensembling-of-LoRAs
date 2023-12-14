# Dynamic Weight Ensembling of LoRAs

This project contains all the code to construct a dynamic weight ensembling of low rank adaptation (LoRA) through the use of a context gating function (formulated as a 2-layer multi-layer perceptron). This project focuses on finetuning T5-large models for two different tasks of haiku and scientific paper title generation.

### Data

The dataset for haiku generation can be found at Hugging Face [here](https://huggingface.co/datasets/statworx/haiku) and the dataset for title generation was originally collected from Arxiv and can be found at Kaggle [here](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts/code).

For convenience, cleaned and pruned datasets for both haiku and title generation are included in this repository at ```haiku_dataset/train.parquet``` and ```arxiv_dataset/train.parquet```, respectively. Both datasets have a total of 26,707 samples with each sample containing a raw input and a reference target text. The haiku dataset contains keywords to prompt generation as its raw input and the actual haiku as the reference target. The title dataset contains scientific paper abstracts as its raw input and the corresponding title of the paper as the reference target.

### Setting up Environment

Before running any experiments, first you must make sure the environment is setup correctly. Follow these steps to prepare your system for running the experiments:

1. Clone this repository to your local system, including the submodule ContextGatedMOE, as this contains the source code for the final context gated mixture of LoRAs model.

2. Install the necessary libraries from the ```requirements.txt``` file by running the following line of code in the  terminal:

   ```pip install -r requirements.txt```

3. Create a ```Results``` folder in the main folder that you cloned this repository into. Create specific subfolders so that the folder structure looks like the following (. represents the main folder that you cloned this repository into).

- .
   - Results
      - Benchmark_Data
      -  Finetune
         - Models
      - Lora_Finetune
         - Base_Models
         - Models
      - CGM_Lora
         - Base_Model
         - Models

Now you are ready to run some experiments!

### Experiments

Each experiment can be run by a single command in the terminal. Each command should be run only while the present working directory is the main folder you cloned this repository into. Each experiment will generate loss graphs for training/validation (except for the benchmark experiment), CSV files containing generated and reference haikus and titles for a withheld test set, as well as checkpoint PyTorch models during finetuning. Follow the steps below to run the experiments and evaluate the results of all the models:

1. **Benchmark Data**

   Run the following line in the terminal:

   ```python Gen_Dataset_Benchmark.py```

   This will create two CSV files in the ```Results/Benchmark_Data``` folder containing generated and reference haikus and titles, respectively.

2. **Full Finetune**

   Run the following line in the terminal:

   ```python Finetune.py```

   This will create two PyTorch models in the ```Results/Finetune/Models``` folder representing the two fully finetuned models, two CSV files in the ```Results/Finetune``` folder containing generated and reference haikus and titles, and two training/validation loss graphs in the ```Results/Finetune``` folder.

3. **LoRA Finetune**

   Run the following line in the terminal:

   ```python Lora_Finetune.py```

   This will create two PyTorch models in the ```Results/Lora_Finetune/Base_Models``` folder representing the two base pretrained models, two PyTorch models in the ```Results/Lora_Finetune/Models``` folder representing the two LoRA finetuned models, two CSV files in the ```Results/Lora_Finetune``` folder containing generated and reference haikus and titles, and two training/validation loss graphs in the ```Results/Lora_Finetune``` folder.

4. **Context Gated Mixture of LoRA Models (CGM LoRA)**

   Run the following line in the terminal:

   ```python CGM_Lora.py```

   This will create one PyTorch model in the ```Results/CGM_Lora/Base_Model``` folder representing the base pretrained model, one PyTorch model in the ```Results/CGM_Lora/Models``` folder representing the context gated mixture of LoRA model, two CSV files in the ```Results/CGM_Lora``` folder containing generated and reference haikus and titles, and one training/validation loss graphs in the ```Results/CGM_Lora``` folder.

5. **Evaluation**

   Run the following line in the terminal:

   ```python Evaluation.py```

   This will create two CSV files in the ```Results``` folder containing Rouge 1, 2, and L scores for all models for both haiku and title generations, two CSV files in the ```Results``` folder containing BERTScores for all models for both haiku and title generation, and one CSV files in the ```Results``` folder containing a grammatical analysis of the generated haikus that checks for the 5-7-5 syllable structure that makes a haiku.
