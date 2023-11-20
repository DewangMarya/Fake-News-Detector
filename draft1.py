# We're trying a few different approaches for this project and we're trying differet methods and models. That's why we named this as "draft1".

# The dataset that we're planning to use is https://www.kaggle.com/c/fake-news/data


# We're going to need pandas to read the dataset
import pandas as pd
from sklearn.model_selection import train_test_split
# The reason why we need TF_IDF is that it helps in identifying certain terms in a document (in our case a dataset)
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# Here is where we load the dataset we plan to use. We found a suitable one from Kaggle, but we're still looking around
df = pd.read_csv('') #We will be loading it later

# Since the scope of our project is smaller than similar projects, we will be setting limits
max_words = 5000
max_length = 100

tokenizer = Tokenizer(num_words=max_words)
# With this, we are trying to update the vocabulary based on the word frequency
tokenizer.fit_on_texts(df['text'])
# Since neural networks require inputs of the same shape, we need to make sure we have padding
sequences = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(sequences, maxlen=max_len)
y = to_categorical(df['label'])

# Here, we're going to split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# This is where we're trying to build the model
model = Sequential()
model.add(Embedding(max_words, 50, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# Softmax is a mathematical function that takes a vector of real numbers as input and transforms it into a probability  distribution
# Reference - https://deepai.org/machine-learning-glossary-and-terms/softmax-layer
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# This is where we're trying to train the model
batch_size = 32
epochs = 10
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# Here, we try to evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Make predictions on new data
new_data = ["The news headline (fake or not)"]
new_data_sequences = tokenizer.texts_to_sequences(new_data)
new_data_padded = pad_sequences(new_data_sequences, maxlen=max_len)
predictions = model.predict(new_data_padded)

# This is to print the predicted class probabilities
print(f'Predicted Probabilities: {predictions}')

# This is to convert predicted probabilities to class labels (fake or true)
predicted_classes = [1 if prob[1] > prob[0] else 0 for prob in predictions]
print(f'Predicted Classes: {predicted_classes}')

