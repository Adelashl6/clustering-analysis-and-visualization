import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import words
from collections import Counter
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import re

stops = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
words = set(words.words())
def load(args):
    data = pd.read_csv(args.data_path)
    labels = pd.read_csv(args.label_path)
    indexes = labels['index'].values.tolist()
    data = data[data['index'].isin(indexes)]
    category_distrib = labels['label'].value_counts()

    plt.figure(figsize=(16, 6))
    ax = sns.barplot(x=category_distrib.index, y=category_distrib.values)
    ax.set_title("Ukraine News Frame Distribution", fontsize="xx-large", y=1.05)
    ax.set_xlabel("Frames", fontsize="x-large")
    ax.set_ylabel("Counts", fontsize="x-large")
    plt.savefig('./ukraine_news/distribution/ukraine_news_frame_distribution.png')
    plt.show()

    # plot distribution of predicted frame clusters
    cluster_distrib = labels['cluster'].value_counts()
    plt.figure(figsize=(16, 6))
    ax = sns.barplot(x=cluster_distrib.index, y=cluster_distrib.values)
    ax.set_title("Ukraine News Predicted Cluster Distribution", fontsize="xx-large", y=1.05)
    ax.set_xlabel("Clusters", fontsize="x-large")
    ax.set_ylabel("Counts", fontsize="x-large")
    plt.savefig('./ukraine_news/distribution/ukraine_news_cluster_distribution.png')
    plt.show()

    return data, labels


def word_tokenize(text):
    # Splits tokens in the text
    tokenized_text = re.findall(r"[~\.,!?;’\":\'؟،؛»«…‘-“”]|[\w]+", text)
    tokenized_text_proc = [token.lower() for token in tokenized_text  # lower-case words
                               if (not token.lower() in stops)  # Remove stopwords
                               and (not len(token) <= 2)  # Remove short tokens
                               and (not re.match("\d+", token))
                               and (token.lower() in words or not token.isalpha())]  # Remove Digits
    # Normalize
    tokenized_text_proc = [lemmatizer.lemmatize(token) for token in tokenized_text_proc]
    return tokenized_text_proc


def preprocess(data):
    # Apply word_tokenize function on each text row
    data['tokens'] = data['sentence'].apply(word_tokenize)

    # Count the most mentioned words
    all_vocab = [token for sublist in data['tokens'].values for token in sublist]
    all_vocab_freq = Counter(all_vocab)
    all_vocab_freq_sorted = sorted(all_vocab_freq.items(), key=lambda x: x[1], reverse=True)
    vocab_df = pd.DataFrame(all_vocab_freq_sorted, columns=["Token", "Frequency"])

    all_vocab = list(set(all_vocab))
    # Plot
    plt.figure(figsize=(16, 6))
    ax = sns.barplot(data=vocab_df.iloc[:50],
                     x="Token", y="Frequency")
    ax.set_title("Total Vocab Frequency", fontsize='xx-large', y=1.05)
    plt.show()
    data.to_csv('./ukraine_news/token.csv', index=False)
    return data, all_vocab


def get_important_feature(cv, matrix, labels):
    clf = LogisticRegression(random_state=0, max_iter=1000, C=10)
    clf.fit(matrix, labels.values)
    print(clf.score(matrix, labels.values))
    sorted_feature_weight_idxes = np.argsort(clf.coef_).squeeze(0)
    sorted_feature_weight_idxes = sorted_feature_weight_idxes[::-1]
    most_important_features = np.take_along_axis(np.array(cv.get_feature_names()), sorted_feature_weight_idxes, axis=0)
    most_important_weights = np.take_along_axis(np.array(clf.coef_).squeeze(0), sorted_feature_weight_idxes, axis=0)
    print(list(zip(most_important_features[:100], most_important_weights))[:100])

    df_feature_weight = pd.DataFrame({'Feature': most_important_features[:100], 'Weight': most_important_weights[:100]})
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.barplot(x="Feature", y="Weight", data=df_feature_weight)
    plt.xticks(rotation=-45, ha="left")
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.title(f'Highest Weight Features in Cluster 4', fontsize='xx-large')
    plt.xlabel('Feature', fontsize=18)
    plt.ylabel('Weight', fontsize=18)
    plt.savefig('./ukraine_news/cluster_4.png')
    plt.show()
    df_feature_weight.to_csv('./ukraine_news/cluster_4.csv', index=0)

def main(args):
    data, labels = load(args)
    data, vocab = preprocess(data)
    cv = TfidfVectorizer(lowercase=False, tokenizer=lambda x: x, vocabulary=vocab)
    cv_matrix = cv.fit_transform(data['tokens'])

    labels = labels['cluster'].map({0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0})
    print((labels==1).sum())
    get_important_feature(cv, cv_matrix, labels)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='find discriminative words for each cluster by logistic regression')
    parser.add_argument('--data_path', default='token.csv', type=str)
    parser.add_argument('--label_path', default='kmeans.csv', type=str)
    args = parser.parse_args()
    main(args)
