import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary resources from NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Input text
text = "Let's follow our great helmsman !"

# Tokenize text into individual words
tokens = word_tokenize(text)

# Perform stemming on each token
stemmed_tokens = [stemmer.stem(token) for token in tokens]

# Perform lemmatization on each token
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

# Join tokens back into text
stemmed_text = ' '.join(stemmed_tokens)
lemmatized_text = ' '.join(lemmatized_tokens)

print("Stemmed text: ", stemmed_text)
print("Lemmatized text: ", lemmatized_text)