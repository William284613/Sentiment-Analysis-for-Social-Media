##### utilities
import re
import numpy as np
import pandas as pd
import nltk
nltk.download('wordnet')
# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# nltk
from nltk.stem import WordNetLemmatizer
# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

#importing datasets  
data_set= pd.read_csv("reddit_vm.csv")  

# Extract year and month of the data
data_set['year'] = pd.DatetimeIndex(data_set['timestamp']).year
data_set['month'] = pd.DatetimeIndex(data_set['timestamp']).month

# Create a count plot
sns.countplot(x='year', data=data_set)
plt.title('Number of posts and comments by year')
plt.show()

sns.countplot(x='month', data=data_set.loc[data_set['year'] == 2019])
plt.title('Number of posts and comments by month (2019)')
plt.show()

data_set.drop(columns=['id', 'url', 'created'], inplace=True)
print(data_set.head())

# convert timestamp from object to datetime
data_set['timestamp'] = pd.to_datetime(data_set['timestamp'], format='%m/%d/%Y %H:%M')
print(data_set.info())

# Pre-processing 
def preprocess(sentence):
    """
    Convert text to lowercase
    Remove website link, special characters, newline character and extra spaces
    """
    sentence = str(sentence)
    sentence = sentence.lower()
    # Remove website links
    rem_url = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sentence)
    
    # Remove special characters
    rem_spe_chr = re.sub(r"[-()\"#/@;:<>\[\]{}`+=~|.!?,]", "", rem_url)
    
    # Remove newline characters
    rem_nl = re.sub(r'\n', ' ', rem_spe_chr)
    
    # Remove numbers
    rem_num = re.sub(r'[0-9]+', '', rem_nl)
    
    # Remove extra spaces
    rem_space = re.sub(r'\s\s+', ' ', rem_num)
    return rem_space

# Load spacy model
nlp = spacy.load('en_core_web_sm')

# Add stopwords
STOP_WORDS.update(['vaccine', 'vaccination', 'vaccinate', 'vaccinated', 'use', 'people', 'person', 'like', 'think', 'know', 'case', 'want',
                   'mean', 'find', 'read', 'point'])

def lemmatize(sentence):
    # Create Doc object and disable components we don't need (for efficiency)
    doc = nlp(sentence, disable=['ner', 'parser'])

    # Generate lemmatized tokens
    lemmas = [token.lemma_ for token in doc]

    # Remove stopwords and non-alphabetic tokens
    alp_lemmas = [lemma for lemma in lemmas 
                  if lemma.isalpha() and lemma not in STOP_WORDS] 

    return ' '.join(alp_lemmas)

# Preprocess and lemmatize the title (non-comment) and body
data_set.loc[data_set['title'] != 'Comment', 'title'] = data_set.loc[data_set['title'] != 'Comment', 'title'].apply(preprocess).apply(lemmatize)
data_set.loc[~data_set['body'].isnull(), 'body'] = data_set.loc[~data_set['body'].isnull(), 'body'].apply(preprocess).apply(lemmatize)

# Identify null values and remove them
data_set = data_set.dropna()
# Reset the index to the default integer index
data_set = data_set.reset_index(drop=True)
print(data_set.head(10)) 

# Tokenization 
data_set['body'] = data_set['body'].apply(lambda x: x.split())

# Stemming
ps = PorterStemmer()
data_set['body'] = data_set['body'].apply(lambda x: [ps.stem(i) for i in x])

# Lemmatizer 
lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data
data_set['body'] = data_set['body'].apply(lambda x: lemmatizer_on_text(x))
print(data_set['body'].head())


# In this, we have set 0 as the score for neutral sentiment.
# Once the score is above 0, sentiment is Positive i.e. People support COVID Vaccination
# If the score is below 0, sentiment is Negative i.e. People are against COVID Vaccination (Anti Vaxxers)
def assign_label(row):
    if row['score'] < 0:
        return 0
    elif row['score'] == 0:
        return 1
    else:
        return 2
    
data_set['label'] = data_set.apply(assign_label, axis=1)
print(data_set.head(10))

# Wordcloud for positive, neutral and negative body 
# Separate data into positive, neutral, and negative sentiments
positive_data = data_set[data_set['label'] == 2]['body']
neutral_data = data_set[data_set['label'] == 1]['body']
negative_data = data_set[data_set['label'] == 0]['body']

# Function to generate and plot word clouds
def generate_and_plot_wordcloud(data, title):
    text = ' '.join(data.apply(lambda x: ' '.join(x)))
    wordcloud = WordCloud(width=800, height=400, random_state=42, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# Generate and plot word clouds for positive, neutral, and negative sentiments
generate_and_plot_wordcloud(positive_data, 'Positive Sentiment Word Cloud')
generate_and_plot_wordcloud(neutral_data, 'Neutral Sentiment Word Cloud')
generate_and_plot_wordcloud(negative_data, 'Negative Sentiment Word Cloud')

#Extracting Independent and dependent Variable  
X = data_set['body'].apply(lambda x: ' '.join(x))
y = data_set['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the training data
X_train = vectorizer.fit_transform(X_train)

# Transform the test data
X_test= vectorizer.transform(X_test) 

def model_Evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))

# Model Building 
# Model 1 - Naive Bayes 
BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel)
y_pred1 = BNBmodel.predict(X_test)

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred1)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Model 2 - SVM (Support Vector Machine)
SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)
model_Evaluate(SVCmodel)
y_pred2 = SVCmodel.predict(X_test) 

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred2)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Model 3 - Logistic Regression 
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel)
y_pred3 = LRmodel.predict(X_test)

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred3)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

