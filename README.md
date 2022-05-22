# Summary
### The main problem of traditional TF-IDF is it doesn't take in consideration of intrinsic nature of document. All the words are given equal importance whereas words appearing at different part have more importance in explaining the contents of the documents.
## Ex. 
![Document_Example](/assets/LR.png)
#### We can see that the heading of the document have "Logistic Regression" which explains that the complete document explains the mentioned concept and thus have more importance in explaining the importance of the document.
#### All the other words in the corpus although might have same word but can be just a reference to a concept rather than having complete information on it.


# Procedure applied in the project
![Document_Example](/assets/Picture1.png)


# Input
### Input should be a dataframe with following columns labeled as below
![Document_Example](/assets/Head.png)
## The above concept can be applied to any layout of documents according to your corpus to attain weighted TF-IDF

# Training and TF-IDF Matrix
## Also the weights can be adjusted to attain the best results.
```R
Vectorizer = TFIDF(df,title_weight = 5,title_gram_range = 2, appendix_gram_range =2, text_gram_range =0,
                 appendix_weight = 3,unigram_weight = 1, bigram_weight = 4, tri_gram_weight = 10)
tf_idf = Vectorizer.get_tfidf()
Vectorizer.save_weights()
```
### Above methods will create 2 files as below:
![Document_Example](/assets/Files.png)

# Load files & Intialize
```R
sparse_matrix = sp.load_npz(os.path.join('','tf_idf.npz'))
with open(os.path.join('','features.txt')) as f:
    data = f.read()
features_dict = json.loads(data)
```

# Retrive documents
```R
def get_similar_articles(q,tf_idf,features_dict,courses):
    vect = CountVectorizer(ngram_range=(1,3))
    vect.fit([q.lower()])
    vect = Counter({w:1 for w in vect.get_feature_names() if w in features_dict.keys()})
    vect.update(features_dict)
    q_vec = np.array(list(({k:vect[k] for k in sorted(vect.keys())}).values()),dtype = 'float64')
    # print
    query_matrix = mb.repmat(q_vec,tf_idf.shape[0],1)
    scores = np.sum(tf_idf*query_matrix,axis =1)/(np.linalg.norm(tf_idf,axis=1)/np.linalg.norm(query_matrix,axis=1))
    similar_scores = Counter({i: score for i,score in enumerate(list(scores))})
    print()
    # sim_sorted = sorted(similar_scores.items(), key=lambda x: x[1], reverse=True)
    sim_sorted = similar_scores.most_common()
    top_10_docs = dict(list(sim_sorted)[:10])
   
    return(top_10_docs) 
# Call the function
temp = get_similar_articles(q1, sparse_matrix,features_dict,df)
```
