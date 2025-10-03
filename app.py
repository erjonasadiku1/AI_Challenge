import pickle
from flask import Flask, request, render_template_string

# Create a Flask app instance
app = Flask(__name__)

# Load the TF-IDF vectorizer we trained earlier
with open("vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Load the trained Logistic Regression model
with open("first_model.pkl", "rb") as f:  
    logistic_model = pickle.load(f)

# Load the label encoder to map numeric labels back to category names
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Function to predict clothing category from a review
def predict_clothing_category(text, model, vectorizer, encoder):
    """
    Takes a text review, converts it into numeric features using TF-IDF,
    predicts the category using the trained model, and returns the
    human-readable category name.
    """
    features = vectorizer.transform([text])
    predicted_label = model.predict(features)[0]
    category = encoder.inverse_transform([predicted_label])[0]
    return category

# The main page of our app where users can input reviews
@app.route("/", methods=["GET", "POST"])
def home():
    predicted_category = ""  
    
    # If the form is submitted
    if request.method == "POST":
        review_text = request.form.get("sentence")  # Get the text input from the user
        if review_text:
            # Predict the category
            predicted_category = predict_clothing_category(
                review_text, logistic_model, tfidf_vectorizer, label_encoder
            )
    
    # Render a simple HTML form and show prediction if available
    return render_template_string("""
        <h2>Women's Clothing Category Predictor</h2>
        <form method="POST">
            <label>Enter a product review:</label><br>
            <input type="text" name="sentence" size="50">
            <input type="submit" value="Predict">
        </form>
        {% if predicted_category %}
            <h3>Predicted category: {{ predicted_category }}</h3>
        {% endif %}
    """, predicted_category=predicted_category)

# Run the app 
if __name__ == "__main__":
    app.run(debug=True)