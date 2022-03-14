'''
Loading Gensim and nltk libraries
'''
import pandas as pd
# pip install gensim
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
np.random.seed(400)
import nltk
import pickle
import argparse

nltk.download('wordnet')
stemmer = SnowballStemmer("english")


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# Tokenize and lemmatize
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))

    return result


def main(args):
    docs = pd.read_csv(args.blog_path)
    disinfo = pd.read_csv(args.cluster_path)

    N = disinfo['tsne_clusters'].values.max()
    keywords = []
    for cluster_idx in range(N):
        processed_docs = []
        subset = disinfo[disinfo['tsne_clusters'] == cluster_idx]
        ids_list = subset['id'].values.tolist()
        for id in ids_list:
            text = docs[docs['id'].astype(int) == id]['post_clean'].values[0]
            processed_docs.append(preprocess(text))

        '''
        Create a dictionary from 'processed_docs' containing the number of times a word appears 
        in the training set using gensim.corpora.Dictionary and call it 'dictionary'
        '''
        dictionary = gensim.corpora.Dictionary(processed_docs)
        dictionary.filter_extremes(no_below=3, no_above=0.2, keep_n=100000)

        '''
        Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'
        '''

        '''
        Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
        words and how many times those words appear. Save this to 'bow_corpus'
        '''
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        '''
        Preview BOW for our sample preprocessed document
        '''

        # example
        '''
        document_num = 20
        bow_doc_x = bow_corpus[document_num]

        for i in range(len(bow_doc_x)):
            print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0],
                                                             dictionary[bow_doc_x[i][0]],
                                                          bow_doc_x[i][1]))
        '''
        lda_model = gensim.models.LdaMulticore(bow_corpus,
                                               num_topics=1,
                                               id2word=dictionary,
                                               passes=20,
                                               workers=2)
        '''
        For each topic, we will explore the words occuring in that topic and its relative weight
        '''

        for idx, topic in lda_model.print_topics(num_words=15):
            print("cluster: {} \nTopic: {} \nWords: {}".format(cluster_idx, idx, topic))
            print("\n")
            keywords.append(topic)

    df = pd.DataFrame({'keywords': keywords})
    df.to_csv(args.save_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='apply t-SNE for dimension reduction and '
                                                 'visualize the data by K-means clustering')
    parser.add_argument('--cluster_path', type=str, default='./ukraine_blog_new/result/tsne_propaganda.csv',
                        help='path to clsutert result')
    parser.add_argument('--blog_path', type=str, default='./ukraine_blog_new/Propaganda_head_blogs.csv',
                        help='path to blog texts')
    parser.add_argument('--save_path',  type=str, default='./ukraine_blog_new/analysis/topic_words/keywords_propaganda.csv',
                        help='path to save topic words')

    args = parser.parse_args()
    main(args)

