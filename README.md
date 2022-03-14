# clustering-analysis-and-visualization
HDBSCAN for clustering Bert-based blog embeddings and extract topic words as well as head actors cluster by cluster

## Clustering and visualization
We run the clustering method HDBSCAN or KMeans on bert-based blog embeddings and use T-SNE to reduce the feature dimension for visualization. \
```python visualize.py --data_path PATH_TO_EMBBEDING_FILE --result_path PATH_TO_SAVE_RESULT --n_component EMBEDDING_DIM_FOR_T-SNE```

## Analysis
### topic words
Extract topic words for each cluster by LDA. \
```python visualize.py --data_path PATH_TO_EMBBEDING_FILE --result_path PATH_TO_SAVE_RESULT --n_component EMBEDDING_DIM_FOR_T-SNE```

### head actors
Given an pre-coded actor lists, obtain key actors with most frequently occurrence for each cluster. \
```python visualize.py --data_path PATH_TO_EMBBEDING_FILE --result_path PATH_TO_SAVE_RESULT --n_component EMBEDDING_DIM_FOR_T-SNE```


### discrminative words
Find discriminative words for each cluster by logistic regression. \
```python visualize.py --data_path PATH_TO_EMBBEDING_FILE --result_path PATH_TO_SAVE_RESULT --n_component EMBEDDING_DIM_FOR_T-SNE```

   
