import numpy as np

from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator
from deeppavlov.skills.odqa.encoders.use_encoder import USEEncoder

iterator = SQLiteDataIterator(load_path='/media/olga/Data/projects/DeepPavlov/download/general_electrics/ge_book.db',
                              shuffle=False)
encoder = USEEncoder()
SAVE_PATH = '/media/olga/Data/projects/DeepPavlov/download/general_electrics/use_vectors'

all_vectors = []
all_lens = []

for docs, _ in iterator.gen_batches(batch_size=100):
    batch_vectors = encoder(docs)
    all_vectors.append(batch_vectors)
    # all_lens.append(batch_lens)

#print(type(all_vectors))
#print(all_vectors.shape)
stacked = np.concatenate(all_vectors, axis=0)
np.save(SAVE_PATH, stacked)
# a = np.load('/media/olga/Data/projects/DeepPavlov/download/odqa/chunk_vectors.npy')
# print(a)
