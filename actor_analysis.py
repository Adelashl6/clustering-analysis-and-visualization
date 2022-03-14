import pandas as pd
import json
import time
import nltk
nltk.download('punkt')
import spacy
from spacy.pipeline import EntityRuler
import argparse
import os


def populate_actors(id_list, docs, actor_list, ukraine_list, cluster):
    current_time= time.time()
    start=current_time

    nlp = spacy.load('en_core_web_sm')

    ruler = EntityRuler(nlp)
    ruler.add_patterns(actor_list['actor_list'])
    nlp.add_pipe(ruler, before='ner')

    ukraine = {}

    for id in id_list:
        text = docs[docs['id'].astype(int) == id]['post_clean'].values[0]
        doc = nlp(text)
        include_ents = ['PERSON', 'NORP', 'GPE', 'ORG']
        for ent in doc.ents:
            if (ent.label_ in include_ents):
                key = ent.text.lower()
                exists = ukraine_list[ukraine_list['Named_Entity'] == key]
                # print("exists::",exists.shape," ent.text.lower()::",ent.text.lower())
                if exists.shape[0] > 0:
                    if key in ukraine.keys():
                        ukraine.update({key: ukraine[key] + 1})
                    else:
                        ukraine[key] = 1
    arr = []
    if len(ukraine) > 0:
        for actor in ukraine:
            arr.append({'actor': actor, 'freq': ukraine[actor]})

    arr = pd.DataFrame(arr).sort_values(['freq'], ascending=False)
    print('time taken for cluster %d: %f ' %(cluster, time.time() - start))
    return arr


def main(args):
    start_time = time.time()

    print('EMFD Start time:', start_time)

    actor_list = json.load(open(args.actor_path, 'r'))
    ukraine_list = pd.read_csv(args.ukraine_path)


    current_time = time.time()
    file_time = current_time - start_time
    print('file imports:', file_time)

    docs = pd.read_csv(args.blog_path)
    result = pd.read_csv(args.cluster_path)
    N = result['tsne_clusters'].values.max()
    for i in range(N):
        id_list = result[result['tsne_clusters'] == i]['id'].values.tolist()
        arr = populate_actors(id_list, docs, actor_list, ukraine_list, i)
        arr.to_csv(os.path.join(args.save_path, 'propaganda_{}.csv'.format(str(i))), index=False)
    print('total time taken', time.time() - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='apply t-SNE for dimension reduction and '
                                                 'visualize the data by K-means clustering')
    parser.add_argument('--actor_path', type=str, default='./blog_new/head_actor/coded_actor_list.json',
                        help='path to cluster result')
    parser.add_argument('--ukraine_path', type=str,
                        default='./blog_new/head_actor/mix_set_camp_coded_entities_includes_additional_set_v4.csv',
                        help='path to blog texts')
    parser.add_argument('--cluster_path', type=str,
                        default='./blog_new/result/tsne_propaganda.csv',
                        help='path to cluster result')
    parser.add_argument('--blog_path', type=str, default='./blog_new/Propaganda_head_blogs.csv', help='path to blog texts')
    parser.add_argument('--save_path', type=str, default='./blog_new/analysis/head_actors')
    args = parser.parse_args()

    main(args)

