# Sentiment Analysis on Amazon Product Reviews

This is my second machine learning project. I used real Amazon product reviews to build a model that can automatically detect if a review is positive, negative or neutral. This type of project is called NLP which means Natural Language Processing.

---

## Why I Did This Project

I wanted to learn how machine learning works with text data. In the house price project I worked with numbers. In this project I wanted to try something different — working with actual sentences and words. Sentiment analysis is also very useful in real life because companies use it to understand what customers think about their products.

---

## Dataset

I used the Amazon Fine Food Reviews dataset from Kaggle.

Link: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

- Total reviews in dataset: 568,454
- I used a sample of 10,000 reviews to make the code run faster
- Each review has a Score from 1 to 5 stars

I converted the star ratings to sentiment labels like this:
- 4 or 5 stars → Positive (1)
- 3 stars → Neutral (0)
- 1 or 2 stars → Negative (-1)

---

## Steps I Followed

**Step 1 — Load the dataset**
I downloaded the dataset using kagglehub and loaded it with pandas.

**Step 2 — Convert ratings to labels**
The dataset has star ratings (1-5). I converted them to three categories: positive, neutral and negative.

**Step 3 — Clean the text**
I converted all text to lowercase and removed punctuation and numbers using regex. This is important so that "Amazing!" and "amazing" are treated as the same word.

**Step 4 — Remove stopwords**
Stopwords are small common words like "the", "is", "i", "and". These words appear in every review whether it is good or bad. So they dont help the model learn. I removed them using a manual list.

**Step 5 — TF-IDF Vectorization**
Machine learning models cannot read text. They only understand numbers. I used TF-IDF to convert each review into a row of numbers. Words that appear a lot in one review but not in others get a high score.

**Step 6 — Train 3 models**
- Logistic Regression
- Naive Bayes
- SVM (Support Vector Machine)

**Step 7 — Evaluate and compare**
I compared all 3 models using accuracy score. I also made a confusion matrix to see where the model makes mistakes.

---

## Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 0.87 |
| Naive Bayes | 0.84 |
| SVM | 0.88 |

SVM gave the best accuracy of 88% on the test data.

Note: These results are with the full 10,000 review sample. If you run with the small manual dataset the accuracy will be lower because there are only 40 reviews.

---

## Plots in This Project

1. Sentiment distribution in the dataset
2. Model accuracy comparison bar chart
3. Confusion matrix for best model

---

## What I Learned

- What NLP means and how text preprocessing works
- How to clean text using regex
- What stopwords are and why we remove them
- How TF-IDF converts text to numbers
- Difference between Logistic Regression, Naive Bayes and SVM for text
- How to read a confusion matrix

---

## How to Run

**Google Colab (recommended):**

Step 1 — Install kagglehub
```
pip install kagglehub
```

Step 2 — Upload your kaggle.json API key
```python
from google.colab import files
files.upload()
```

Step 3 — Move the key to correct folder
```python
import os
os.makedirs('/root/.config/kaggle', exist_ok=True)
os.rename('kaggle.json', '/root/.config/kaggle/kaggle.json')
```

Step 4 — Run the code
```
python sentiment_analysis.py
```

**VS Code:**
```
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
python sentiment_analysis.py
```

---

## Files

- `sentiment_analysis.py` — main Python code
- `README.md` — project description

---

## Author

**Muhammad Ahmad**
BS Information Technology — University of Sargodha
ahmadsheikh.bu@gmail.com
linkedin.com/in/ahmadsheikh21# sentiment-analysis-product-reviews
Classifying Amazon product reviews as positive, negative or neutral using NLP and machine learning in Python.
