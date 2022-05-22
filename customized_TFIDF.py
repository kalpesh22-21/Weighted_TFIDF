import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import scipy.sparse as sp
from numpy.linalg import norm
import numpy.matlib as mb
import json

class TFIDF(object):

    def __init__(self,df,title_weight = 5,title_gram_range = 2, appendix_gram_range =2, text_gram_range =0,
                 appendix_weight = 3,unigram_weight = 1, bigram_weight = 4, tri_gram_weight = 10):
        
        self.corpus = df.text
        self.corpus_title = df.Title
        self.corpus_appendix = df.Appendix
        
        self.title_gram_range = title_gram_range
        self.appendix_gram_range = appendix_gram_range
        self.text_gram_range = text_gram_range
        
        self.gram_weights = {}
        self.gram_weights[1] = unigram_weight
        self.gram_weights[2] = bigram_weight
        self.gram_weights[3] = tri_gram_weight
        
        self.title_weight = title_weight
        self.appendix_weight = appendix_weight
        self.text_weight = 1
        
###### Vector Functions
    def preprocessing_text(self):
        n_c = np.vectorize(self.__clean__)
        self.norm_corpus_title= n_c(self.corpus_title)
        self.norm_corpus_appendix= n_c(self.corpus_appendix)
        self.norm_corpus = n_c(self.corpus)
        del(self.corpus_title,self.corpus_appendix,self.corpus)

    def build_corp_vect(self,title,appendix,corp):   
        document = title + '|' + appendix + '|'+ corp
        words = document.split('|')
        # words = np.lib.pad(words, ((0,self.N-len(words))), 'constant', constant_values= '')
        words = np.array(words,dtype= object)
        return(words)
######
    
    def __clean__(self,d):
        #Removing Stop Words
        stop_words = nltk.corpus.stopwords.words('english')
        d = re.sub(r'[^a-zA-Z0-9\s]', '', str(d), re.I|re.A)
        d = d.lower().strip()
        tks = nltk.word_tokenize(d)
        return(' '.join([t for t in tks if t not in stop_words]))
        
    def gram_dict(self,packet):
        try:
            [[text,i,weight]] = [[text,i,weight] for ([text,i,weight],) in packet]
            vectorizer = CountVectorizer(ngram_range=(i,i))
            vector = vectorizer.fit_transform(text)
            tf_dict = pd.DataFrame(vector.toarray()*weight*self.gram_weights[i],columns = list(vectorizer.get_feature_names_out())).to_dict()
            return(tf_dict)
        except:
            pass
    
    
    def n_tf_builder(self,title,appendix,text):
        tf_vect = np.vectorize(self.gram_dict)
        gram_packets = []
        for i in range(self.title_gram_range):
            gram_packets.append(zip([[title,i+1,self.title_weight]]))
            
        for i in range(self.appendix_gram_range):
            gram_packets.append(zip([[appendix,i+1,self.appendix_weight]]))

        for i in range(self.text_gram_range):
            gram_packets.append(zip([[text,i+1,self.text_weight]]))
            
        dicts = tf_vect(gram_packets)
        tf = {}
        for dict_ in dicts:
            if dict_ != None:
                tf.update(dict_)
        tf = pd.DataFrame.from_dict(tf)
        tf = tf[sorted(tf.columns)]
        return(tf)
        
    def cal_tf(self):
        try:
            tf = self.n_tf_builder(self.norm_corpus_title,self.norm_corpus_appendix,self.norm_corpus)
        except:
            tf = self.n_tf_builder(self.corpus_title,self.corpus_appendix,self.corpus)
        self.tf = tf
        self.sum_vect = norm(self.tf , axis=0)
        self.features_dict = { w:0 for w in self.tf.columns}
        # self.tf = pd.DataFrame(list(tf),columns = self.features_dict)
        
    def cal_df(self):
        df = np.diff(sp.csc_matrix(self.tf, copy=True).indptr)
        df = 1 + df
        self.df = df
        
    def cal_idf(self):
        N = 1 + len(self.corpus)
        idf = (1.0 + np.log(float(N) / self.df)) 
        idf_d = sp.spdiags(idf, diags= 0, m=len(df), n= len(df)).todense()
        del(self.df)
        self.idf = idf
#         self.idf_d = idf_d

    def get_tfidf(self):
        self.cal_tf()
        self.cal_df()
        self.cal_idf()
        tf = np.array(self.tf, dtype='float64')
        tfidf = tf * self.idf/self.sum_vect
        norms = norm(tfidf , axis=1)
        tfidf = pd.DataFrame(tfidf / norms[:,None],columns=sorted(self.tf.columns))
        self.tfidf = sp.csr_matrix(tfidf.values)
        return (tfidf)
    
    
    def vectorize(self,query):
        vect = CountVectorizer(ngram_range=(1,3))
        vect.fit(query)
        vect = Counter({w:1 for w in vect.get_feature_names() if w in self.features_dict})
        vect.update(self.features_dict)
        return(np.array(list(({k:vect[k] for k in sorted(vect.keys())}).values()),dtype = 'float64'))
    
    
    def similarity(self,query_vector,tfidf_csr = None):
        if tfidf_csr == None:
            scores = self.tfidf.dot(q_vec)/(sp.linalg.norm(self.tfidf,axis=1)/np.linalg.norm(q_vec,axis=0))
        else:
            scores = tfidf_csr.dot(q_vec)/(sp.linalg.norm(tfidf_csr,axis=1)/np.linalg.norm(q_vec,axis=0))
            
        similar_scores = {i: score for i,score in enumerate(list(scores))}
        similar_scores = sorted(similar_scores.items(), key=lambda x: x[1], reverse=True)
        return(similar_scores)
    
    
    def save_weights(self,file_path = ''):
        sp.save_npz(os.path.join(file_path,'tf_idf.npz'), self.tfidf)
        json.dump(self.features_dict,open(os.path.join(file_path,'features.txt'), 'w'))

        
    def load_tf_idf(self,file_path):
        self.tfidf = sp.load_npz(os.path.join(file_path,'tf_idf.npz'))
        with open(os.path.join(file_path,'features.txt')) as f:
            data = f.read()
        self.features_dict = json.loads(data)
        
        
    def update(self,tf,New_Title,New_Appendix,New_Corpus):
        old_vectors = tf.to_dict()
        old_vocabulary = tf.columns

        new_vectors = self.n_tf_builder(New_Title,New_Appendix,New_Corpus)
        new_vectors = new_vectors.set_index(np.arange(tf.shape[0],new_vectors.shape[0]))
        new_vocabulary = new_vectors.columns
        new_vectors = new_vectors.to_dict()
        
        #update common words
        common = set(new_vocabulary).intersection(old_vocabulary)
        for word in common:
            old_vectors[word].update(new_vectors[word])
            
        # update new words
        new_words = list(set(new_vocabulary) - set(old_vocabulary))
        temp = {i : 0 for i in range(tf.shape[0])}
        for word in new_words:
            old_vocabulary[word] =  temp.update(new_vectors[word])
        
        # update old words
        old_words = list(set(old_vocabulary) - set(new_vocabulary))
        temp = {i : 0 for i in np.arange(tf.shape[0],new_vectors.shape[0])}
        for word in old_words:
            old_vocabulary[word] =  old_vocabulary[word].update(temp)
        
        self.tf = pd.DataFrame.from_dict(old_vocabulary)
        self.get_tfidf()