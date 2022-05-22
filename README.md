# Summary
### The main problem of traditional TF-IDF is it doesn't take in consideration of intrinsic nature of document. All the words are given equal importance whereas words appearing at different part have more importance in explaining the contents of the documents.
## Ex. 
![Document_Example](/assets/LR.png)
#### We can see that the heading of the document have "Logistic Regression" which explains that the complete document explains the mentioned concept and thus have more importance in explaining the importance of the document.
#### All the other words in the corpus although might have same word but can be just a reference to a concept rather than having complete information on it.


### Procedure applied in the project
  ![Document_Example](/assets/Picture1.png)
  1. The cleaned data files are fetched through the pre-processing pipeline to eradicate the unnecessary contextual words that don't aid the document search process.
  2. The cleaned document is then manually split into Title, Table of Contents & Course Content (Chapters, Appendix & References) for further weighting implementation. 
  3. We then generate Unigrams, Bi-Grams and Trigrams for each Title, Table of Content and Content Material. 
  4. These N-grams are assigned the respective weightage to quantify the level of importance in order to fine tune the search grid parameter as follows - 

    Title (Weight) -10   
    Table of Contents (Weight) – 5
    Course Content (Chapters, Appendix & References) Weight - 1
  	The outputs from the N-grams are then combined to form a custom weighted term frequency matrix 
    
  5. This term frequency matrix is then combined with the inverse document frequency to form a TF-IDF vector matrix that ensures the marginalization of words that are repeated too often within and across documents. 
  6. This implementation is then used to compute a cosine similarity score across all documents based on the matrix comparison with the user searched query.
  7. Finally, the model puts out the top-n courses which holds the maximum cosine similarity score out of all the documents to the chatbot API

## The above concept can be applied to any layout of documents according to your corpus to attain weighted TF-IDF
## Also the weights can be adjusted to attain the best results.
