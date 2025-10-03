# Women's Clothing E-Commerce Reviews Classification

This project focuses on analyzing and classifying customer reviews for women's clothing products using machine learning models. The dataset contains textual reviews along with product metadata, and the goal is to predict the category of clothing based on review text and related features.

---

## Dataset

- **Source:** `Womens Clothing E-Commerce Reviews.csv`
- **Rows:** 23,486  
- **Columns:** 11  
- **Content:** Customer reviews, ratings, product categories, and department information.

**Important Notes:**

- Some columns contain missing values, which are handled during preprocessing.  
- Duplicate rows are removed to avoid bias.  
- Classes with very few samples are removed to improve model performance.

---

## Data Preprocessing

1. **Handling missing values:**  
   - Rows with missing target labels (`Class Name`) are dropped.  
   - Missing values in input features (`Title`, `Review Text`, `Division Name`, `Department Name`) are filled with placeholders.

2. **Text normalization:**  
   - Lowercasing  
   - Removing punctuation and extra spaces  
   - Removing stopwords (for TF-IDF vectorization)  
   - Optional stemming using `PorterStemmer`

3. **Encoding categorical data:**  
   - Target labels (`Class Name`) are encoded using **Label Encoding**.  
   - Categorical features (`Division Name`, `Department Name`) are one-hot encoded.

4. **Text feature extraction:**  
   - Reviews are transformed into numeric features using **TF-IDF Vectorization** (top 5000 features).

---

## Exploratory Data Analysis

- Distribution of classes shows imbalance, with **Dresses**, **Blouses**, and **Knits** being the most common.  
- Correlation analysis shows numeric features are mostly independent; the strongest correlation is between `Rating` and `Recommended IND`.  
- Word clouds highlight the most frequent words in reviews.  

---

## Model Training

Two models were trained and evaluated:

1. **Logistic Regression**  
   - Accuracy: ~62%  
   - Performs well for larger, clearer classes.  
   - Struggles with smaller or ambiguous classes due to class imbalance.

2. **Naive Bayes (MultinomialNB)**  
   - Accuracy: ~44%  
   - Performs well only for the most common classes.  
   - Poor performance for smaller or less frequent categories.

**Observations:**
- Class imbalance strongly affects model performance.  
- Potential solutions: oversampling smaller classes, undersampling larger classes, class-weighted models, or using a smaller, balanced dataset.

---

## Confusion Matrix

The confusion matrix for Logistic Regression confirms:

- Strong performance for the most frequent classes.  
- Misclassifications occur primarily in smaller or similar categories.  
- Strategies to improve results include balancing the dataset or adjusting class weights.

---

## Predicting New Reviews

A helper function allows predicting the clothing category for new reviews:

```python
def predict_category(text, model, vectorizer, label_encoder):
    text_tfidf = vectorizer.transform([text])
    pred_label = model.predict(text_tfidf)[0]
    category = label_encoder.inverse_transform([pred_label])[0]
    return category

Example:

user_input = "This dress fits perfectly and looks elegant."
predicted_category = predict_category(user_input, logReg, vectorizer, le)
print(predicted_category)

```

## Saving Models

The following objects are saved for future use:
vectorizer.pkl → TF-IDF vectorizer
logReg.pkl → Trained Logistic Regression model
label_encoder.pkl → Label encoder for translating numeric labels back to categories

```python
import pickle
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(logReg, open("logReg.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

```

## Requirements

Python 3.x

pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, wordcloud

## Conclusion

Logistic Regression outperforms Naive Bayes on this dataset.
Class imbalance and sparse classes are the main challenges.
Future improvements could include more advanced text embeddings (e.g., BERT), balancing the dataset, or using ensemble methods for better accuracy.