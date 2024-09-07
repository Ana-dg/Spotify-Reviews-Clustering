import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import spacy
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

file_path = '/content/reviews.csv'
df = pd.read_csv(file_path)
df.head()
df = df.drop(['Time_submitted', 'Reply','Total_thumbsup'], axis=1)
df.head()
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    return text
df['Review'] = df['Review'].apply(clean_text)
print(df.describe())
print("\n")
print(df.isnull().sum())
df['Rating'].describe()
# Distribution of Ratings
plt.figure(figsize=(8, 6))
sns.countplot(x='Rating', data=df, palette='viridis')
plt.title('Distribution of Ratings')
plt.show()
wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='white').generate(' '.join(df['Review']))
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Word Cloud for Reviews')
plt.show()
# Text Preprocessing and Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000, lowercase=True)
tfidf_matrix = vectorizer.fit_transform(df['Review'])
count_vectorizer = CountVectorizer(max_features=20, stop_words='english')
count_matrix = count_vectorizer.fit_transform(df['Review'])
# Get words
feature_names = count_vectorizer.get_feature_names_out()
term_frequencies = count_matrix.sum(axis=0)
tf_df = pd.DataFrame(data=term_frequencies, columns=feature_names).transpose()
tf_df.columns = ['Term Frequency']
tf_df.sort_values(by='Term Frequency', ascending=False).plot(kind='bar', figsize=(10, 6), color='skyblue')
plt.title('Top 20 Terms by Frequency after Tokenization')
plt.xlabel('Term')
plt.ylabel('Frequency')
plt.show()
custom_stopwords = ['app', 'music', 'spotify', 'songs', 'song', 'play', 'just', 'listen', 'dont','im','playing','just','use','want']
def remove_stopwords(text):
    pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, custom_stopwords)))
    return re.sub(pattern, '', text, flags=re.IGNORECASE)
df['Review'] = df['Review'].apply(remove_stopwords)
ratings = df['Rating'].unique()
ngram_range = (2, 3)
top_n_ngrams = 10
fig, axes = plt.subplots(len(ratings), 1, figsize=(10, 4 * len(ratings)), sharex=True)
ratings = sorted(ratings, reverse=True)

for i, rating in enumerate(ratings):
    reviews_for_rating = df[df['Rating'] == rating]['Review']
    vectorizer_ngram = CountVectorizer(max_features=5000, lowercase=True, ngram_range=ngram_range, stop_words='english')
    ngram_matrix = vectorizer_ngram.fit_transform(reviews_for_rating)
    feature_names_ngram = vectorizer_ngram.get_feature_names_out()
    ngram_df = pd.DataFrame(data=ngram_matrix.toarray(), columns=feature_names_ngram)
    top_ngram_words = ngram_df.sum(axis=0).sort_values(ascending=False).head(top_n_ngrams)
    top_ngram_words.plot(kind='barh', color='skyblue', ax=axes[i])
    axes[i].set_title(f'Top {top_n_ngrams} N-grams for Rating {rating}')
    axes[i].set_xlabel('Frequency')
    axes[i].set_ylabel('N-gram')
plt.show()
# Sample the reviews with ratings 1, 2, 3 into reviews with rating 1
df['Sampled_Rating'] = np.where(df['Rating'].isin([1, 2, 3]), 1, df['Rating'])
# Sample the reviews with ratings 4, 5 into reviews with rating 5
df['Sampled_Rating'] = np.where(df['Rating'].isin([4, 5]), 5, df['Sampled_Rating'])
# Distribution of Ratings
plt.figure(figsize=(8, 6))
sns.countplot(x='Sampled_Rating', data=df, palette='viridis')
plt.title('Distribution of Ratings')
plt.show()
#Lemmatization
nlp = spacy.load("en_core_web_sm")
def lemmatize(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])
df['lemmatized_text'] = df['Review'].apply(lemmatize)
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, lowercase=True)
tfidf_matrix = vectorizer.fit_transform(df['lemmatized_text'])
#LDA clustering
true_labels = df['Rating']
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, true_labels, test_size=0.2, random_state=42)
num_topics = 5
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
topic_assignments_lda = lda.fit_transform(X_train)
topic_assignments_lda_test = lda.transform(X_test)
ari_lda = adjusted_rand_score(y_test, topic_assignments_lda_test.argmax(axis=1))
print(f"LDA, ARI: {ari_lda}")
#Random search for LDA
param_grid = {
    'n_components': [2, 5, 10],  # Adjust the values as needed
    'learning_decay': [0.5, 0.7, 0.9],
    'max_iter': [20, 50, 100]
}
lda = LatentDirichletAllocation()
grid_search = GridSearchCV(lda, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(tfidf_matrix)
print("Best Parameters:", grid_search.best_params_)
#Accuarcy after gridsearch
best_lda_params = {'learning_decay': 0.7, 'max_iter': 50, 'n_components': 2}
lda = LatentDirichletAllocation(**best_lda_params, random_state=42)
topic_assignments_lda = lda.fit_transform(X_train)
topic_assignments_lda_test = lda.transform(X_test)
ari_lda = adjusted_rand_score(y_test, topic_assignments_lda_test.argmax(axis=1))
print(f"LDA, ARI: {ari_lda}")
#PCA component analysis
tfidf_standardized = StandardScaler().fit_transform(tfidf_matrix.toarray())
pca_full = PCA()
pca_result_full = pca_full.fit_transform(tfidf_standardized)
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
plt.plot(cumulative_explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()
#PCA with 3000 components
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()
num_components_pca = 3000
pca = PCA(n_components=num_components_pca, random_state=42)
X_train_pca = pca.fit_transform(X_train_dense)
X_test_pca = pca.transform(X_test_dense)
#GMM
gmm = GaussianMixture(random_state=42)
topic_assignments_gmm = gmm.fit_predict(X_train_pca)
# Transform X_test using the trained GMM model
topic_assignments_gmm_test = gmm.predict(X_test_pca)
# Evaluate GMM results on the test set
ari_gmm = adjusted_rand_score(y_test, topic_assignments_gmm_test)
print(f"GMM, ARI: {ari_gmm}")
#Optimal number of components
n_components = range(1, 11)
models = [GaussianMixture(n, covariance_type='full', random_state=42).fit(X_train_pca) for n in n_components]
plt.plot(n_components, [model.score(X_train_pca) for model in models], marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Log Likelihood')
plt.title('GMM Elbow Method')
plt.show()
#GridSearch GMM
param_grid = {
    'n_components': [5],
    'covariance_type': ['full', 'tied', 'diag'],
    'max_iter': [50, 100, 200]
}
gmm = GaussianMixture(random_state=42)
grid_search = GridSearchCV(gmm, param_grid, scoring='adjusted_rand_score', cv=5)
grid_search.fit(X_train_pca, y_train)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
best_gmm_model = grid_search.best_estimator_
topic_assignments_gmm_test = best_gmm_model.predict(X_test_pca)
ari_gmm = adjusted_rand_score(y_test, topic_assignments_gmm_test)
print(f"GMM, ARI: {ari_gmm}")
best_gmm_params = {'n_components': 5, 'covariance_type': 'full', 'max_iter': 100}
gmm = GaussianMixture(**best_gmm_params, random_state=42)
topic_assignments_gmm = gmm.fit_predict(X_train_pca)
topic_assignments_gmm_test = gmm.predict(X_test_pca)
ari_gmm = adjusted_rand_score(y_test, topic_assignments_gmm_test)
print(f"GMM with PCA, ARI: {ari_gmm}")
# Random baseline
dummy_model = DummyClassifier(strategy="uniform", random_state=42)
dummy_model.fit(X_train, y_train)
dummy_predictions = dummy_model.predict(X_test)
ari_random = adjusted_rand_score(y_test, dummy_predictions)
print(f"Random Baseline, ARI: {ari_random}")
# Supervised baseline using Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
lr_predictions = lr.predict(X_test)
ari_lr = adjusted_rand_score(y_test, lr_predictions)
print(f"Logistic Regression, ARI: {ari_lr}")
accuracy = accuracy_score(y_test, lr_predictions)
print(f"Accuracy: {accuracy}")