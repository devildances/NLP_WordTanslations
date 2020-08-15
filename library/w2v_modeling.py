from gensim.models import KeyedVectors
import pickle
from library import utils_wt as uwt

def build_w2v_model():
    print('Start building Word2Vec models...')
    en_embeddings = KeyedVectors.load_word2vec_format('RAW/en/GoogleNews-vectors-negative300.bin', binary = True)
    id_embeddings = KeyedVectors.load_word2vec_format('RAW/id/wiki.id.vec')

    en_id_train = uwt.get_dict_en_id('RAW/train_test/en-id.train.txt')
    print('The length of the english to indonesia training dictionary is', len(en_id_train))
    en_id_test = uwt.get_dict_en_id('RAW/train_test/en-id.test.txt')
    print('The length of the english to indonesia test dictionary is', len(en_id_test))

    english_set = set(en_embeddings.vocab)
    indonesia_set = set(id_embeddings.vocab)
    en_embeddings_subset = {}
    id_embeddings_subset = {}
    indonesia_words = set(en_id_train.values())

    for en_word in en_id_train.keys():
        id_word = en_id_train[en_word]
        if id_word in indonesia_set and en_word in english_set:
            en_embeddings_subset[en_word] = en_embeddings[en_word]
            id_embeddings_subset[id_word] = id_embeddings[id_word]


    for en_word in en_id_test.keys():
        id_word = en_id_test[en_word]
        if id_word in indonesia_set and en_word in english_set:
            en_embeddings_subset[en_word] = en_embeddings[en_word]
            id_embeddings_subset[id_word] = id_embeddings[id_word]


    pickle.dump( en_embeddings_subset, open( "model/en_embeddings.p", "wb" ) )
    pickle.dump( id_embeddings_subset, open( "model/id_embeddings.p", "wb" ) )
    print('Models have been built and stored into model directory!')