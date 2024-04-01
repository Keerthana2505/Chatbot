import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import nltk
from nltk.stem import WordNetLemmatizer
import string

# Load the data
data = pd.read_csv('MasterData.csv')

# Preprocess text function
def preprocess_text(text):
    if isinstance(text, str):  # Check if the input is a string
        text = text.lower()
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)
    else:
        return ''  # Return an empty string or handle the case as appropriate


# Preprocess data
lemmatizer = WordNetLemmatizer()
data['patterns'] = data['Value.patterns'].apply(preprocess_text)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['patterns'], data['Value.responses'], test_size=0.2, random_state=42)

# Encode the target variable
label_encoder = LabelEncoder()
label_encoder.fit(y_train)  # Fit encoder on training labels
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)  # Transform test labels using the same encoder
# Check for consistency between training and test labels
unseen_labels = set(y_test) - set(label_encoder.classes_)
if unseen_labels:
    print("Warning: Unseen labels in test set:", unseen_labels)

# Build the ANN model using Keras
model = Sequential()
model.add(Dense(128, input_dim=1, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test, y_test_encoded))

# Predict the outcomes
y_pred = model.predict(X_test)


# Deploy the model in the Flask app
# Flask app implementation
from flask import Flask, request, jsonify,render_template

app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['text']
    input_text = preprocess_text(input_text)
    prediction = model.predict([input_text])
    predicted_class = label_encoder.inverse_transform([prediction.argmax()])
    return jsonify({'predicted_class': predicted_class[0]})

if __name__ == '__main__':
    app.run(debug=True)
