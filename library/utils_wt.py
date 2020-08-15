import pandas as pd
import numpy as np

def get_dict_en_id(file_name):
    """
    This function returns the english to indonesia dictionary given a file where the each column corresponds to a word.
    """
    my_file = pd.read_csv(file_name, delimiter='\t', names=['en','id'])
    etof = {}
    for i in range(len(my_file)):
        en = my_file.loc[i][0]
        idn = my_file.loc[i][1]
        etof[en] = idn

    return etof

def get_matrices(en_id, indonesia_vecs, english_vecs):
    """
    Input:
        en_id: English to Indonesia dictionary
        indonesia_vecs: Indonesia words to their corresponding word embeddings.
        english_vecs: English words to their corresponding word embeddings.
    Output:
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the Indonesia embeddings.
        R: the projection matrix that minimizes the F norm ||X R -Y||^2.
    """
    X_l = list()
    Y_l = list()
    english_set = english_vecs.keys()
    indonesia_set = indonesia_vecs.keys()
    indonesia_words = set(en_id.values())

    for en_word, id_word in en_id.items():

        if id_word in indonesia_set and en_word in english_set:
            en_vec = english_vecs[en_word]
            id_vec = indonesia_vecs[id_word]
            X_l.append(en_vec)
            Y_l.append(id_vec)

    X = np.vstack(X_l)
    Y = np.vstack(Y_l)

    return X, Y

def compute_loss(X, Y, R):
    '''
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the Indonesia embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to Indonesia vector space embeddings.
    Outputs:
        L: a matrix of dimension (m,n) - the value of the loss function for given X, Y and R.
    '''
    m = X.shape[0]
    diff = np.dot(X,R) - Y
    diff_squared = np.square(diff)
    sum_diff_squared = np.sum(diff_squared)
    loss = sum_diff_squared/m
    return loss

def compute_gradient(X, Y, R):
    '''
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the Indonesia embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to Indonesia vector space embeddings.
    Outputs:
        g: a matrix of dimension (n,n) - gradient of the loss function L for given X, Y and R.
    '''
    m = X.shape[0]
    gradient = np.dot(X.T, np.dot(X,R)-Y) * 2/m
    return gradient

def align_embeddings(X, Y, train_steps=100, learning_rate=0.0003):
    '''
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the Indonesia embeddings.
        train_steps: positive int - describes how many steps will gradient descent algorithm do.
        learning_rate: positive float - describes how big steps will  gradient descent algorithm do.
    Outputs:
        R: a matrix of dimension (n,n) - the projection matrix that minimizes the F norm ||X R -Y||^2
    '''
    np.random.seed(129)
    R = np.random.rand(X.shape[1], X.shape[1])

    for i in range(train_steps):
        if i % 25 == 0:
            print(f"loss at iteration {i} is: {compute_loss(X, Y, R):.4f}")
        gradient = compute_gradient(X, Y, R)
        R -= learning_rate * gradient

    return R

def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''
    cos = -10
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot / (norma * normb)

    return cos

def nearest_neighbor(v, candidates, k=1):
    """
    Input:
      - v, the vector that are going find the nearest neighbor for
      - candidates: a set of vectors where will find the neighbors
      - k: top k nearest neighbors to find
    Output:
      - k_idx: the indices of the top k closest vectors in sorted form
    """
    similarity_l = []

    for row in candidates:
        cos_similarity = cosine_similarity(v, row)
        similarity_l.append(cos_similarity)

    sorted_ids = np.argsort(similarity_l)
    k_idx = sorted_ids[-k:]

    return k_idx

def test_vocabulary(X, Y, R):
    '''
    Input:
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the Indonesia embeddings.
        R: the transform matrix which translates word embeddings from
        English to Indonesia word vector space.
    Output:
        accuracy: for the English to Indonesia capitals
    '''
    pred = np.dot(X, R)
    num_correct = 0

    for i in range(len(pred)):
        pred_idx = nearest_neighbor(pred[i], Y)

        if pred_idx == i:
            num_correct += 1

    accuracy = num_correct/pred.shape[0]

    return accuracy