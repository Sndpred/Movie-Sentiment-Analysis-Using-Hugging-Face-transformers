# Movie Sentiment Analysis Using Hugging Face Transformers
This repository contains a project focused on movie review sentiment analysis using Hugging Face Transformers. The core of the project is a Jupyter notebook that guides users through loading and exploring a sentiment dataset, comparing various pre-trained models, fine-tuning a DistilBERT model, and creating a production-ready pipeline for sentiment prediction.

**Project Structure**.

├── environment.txt            # Text file to create and activate the Python environment <br> 
└── Hugging Face Sentiment Analysis.ipynb  # Jupyter Notebook with project code <br> 

Note: The fine-tuned-distilbert-imdb/ folder (containing the saved model) and the results/ folder (for training output and logs) are generated automatically when you run the Hugging Face Sentiment Analysis.ipynb notebook. They are not included in the repository to keep its size manageable.

**Getting Started**

Follow these steps to set up and run the project locally.
Prerequisites

    Python 3.8+

    pip (Python package installer)

    Git (for cloning the repository)

**1. Clone the Repository:**

First, clone this repository to your local machine:
git clone https://github.com/your-username/HuggingFace-Sentiment-Analysis.git
cd HuggingFace-Sentiment-Analysis

**2. Create and Activate Virtual Environment:**

It's highly recommended to use a virtual environment to manage project dependencies.
Assuming your environment.txt file lists the required Python packages, you can create and activate a new environment like this:

**Create a new conda environment (if you use Anaconda/Miniconda)**
conda create -n hf-sentiment python=3.10  # Or your preferred Python version
conda activate hf-sentiment

**OR create a virtualenv (if you prefer venv)**
python3 -m venv hf-sentiment
source hf-sentiment/bin/activate

**3. Install Dependencies:**

Install the necessary Python packages using the environment.txt file:
pip install -r environment.txt

If you don't have an environment.txt file yet, you'll need to create one by listing all the packages used in your Jupyter notebook (e.g., torch, transformers, datasets, pandas, matplotlib, seaborn, scikit-learn).

**4. Run the Jupyter Notebook to Generate Model and Results:**

Launch Jupyter Notebook and open the Hugging Face Sentiment Analysis.ipynb file:
jupyter notebook "Hugging Face Sentiment Analysis.ipynb"

Run all the cells in the notebook. This will:

    Load and explore the IMDB sentiment dataset.

    Compare the performance of several pre-trained sentiment analysis models from Hugging Face.

    Fine-tune a DistilBERT model on the IMDB dataset, which will create the fine-tuned-distilbert-imdb folder.

    Analyze the fine-tuning results.

    Generate training logs and checkpoints in the results folder.

    Test a production-ready sentiment analyzer class.

**Project Overview:**
The Hugging Face Sentiment Analysis.ipynb notebook covers the following key steps:

    Dataset Loading and Exploration: Utilizes the datasets library to load the IMDB dataset and performs basic data analysis and visualization.

    Pre-trained Model Comparison: Evaluates several pre-trained Hugging Face models (cardiffnlp/twitter-roberta-base-sentiment-latest, distilbert-base-uncased-finetuned-sst-2-english, nlptown/bert-base-multilingual-uncased-sentiment) on sample data to compare their speed and prediction formats.

    DistilBERT Fine-tuning: Demonstrates how to fine-tune the distilbert-base-uncased-finetuned-sst-2-english model on a subset of the IMDB dataset using the transformers Trainer API.

    Results Analysis: Compares the performance of the pre-trained and fine-tuned models, highlighting improvements in training and validation loss.

    Production Pipeline: Creates a MovieSentimentAnalyzer class that wraps the fine-tuned model, providing predict and predict_batch methods for easy sentiment inference.

