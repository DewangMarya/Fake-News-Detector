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

#Since the scope of our project is smaller than similar projects, we will be setting limits
max_words = 5000
max_length = 100


