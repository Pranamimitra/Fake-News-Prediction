# Fake News Detection using Logistic Regression

This project implements a Fake News Detection system using the Logistic Regression model in Python. The system aims to classify news articles as real or fake based on the textual content. The project involves various stages of data preprocessing, including stemming and vectorization, to convert the textual data into numerical form suitable for machine learning.

## Dependencies

The project uses the following libraries:

- numpy
- pandas
- re
- nltk (Natural Language Toolkit)
- sklearn (scikit-learn)

## Data Preprocessing

1. **Loading Dataset**: The dataset is loaded into a pandas DataFrame.
2. **Handling Missing Values**: Missing values in the dataset are replaced with empty strings.
3. **Merging Features**: The author name and news title are merged to form the content feature.
4. **Stemming**: Reducing words to their root form using the PorterStemmer.
5. **Vectorization**: Converting textual data into numerical data using TfidfVectorizer.

## Model Training

The dataset is split into training and testing sets. A Logistic Regression model is trained on the training data and evaluated using the test data.

## Evaluation

The model's performance is evaluated using accuracy scores on both the training and test datasets.

## Prediction

The system includes a predictive function to classify new news articles as real or fake.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

```bash
pip install numpy pandas nltk scikit-learn
```

### Running the Project

1. Clone the repository.
2. Ensure you have the dataset `train.csv` in the appropriate directory.
3. Run the script to preprocess the data, train the model, and evaluate its performance.

### Example Usage

```python
python fake_news_detection.py
```

This will load the dataset, preprocess it, train the model, and print the accuracy scores. It will also demonstrate a predictive example by classifying a news article from the test set.
