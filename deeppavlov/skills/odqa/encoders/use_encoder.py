from typing import List, Tuple, Callable

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from nltk.tokenize import sent_tokenize

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.commands.utils import expand_path


@register('use_encoder')
class USEEncoder(Component):
    def __init__(self, sentencize_fn: Callable = sent_tokenize, **kwargs):
        """
        :param top_n: top n sentences to return
        :param return_vectors: return unranged USE vectors instead of sentences
        :param active: when is not active, return all sentences
        """
        self.embed = hub.Module(str(expand_path("general_electrics/hub")))
        self.session = tf.Session(config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=0.4,
                allow_growth=False
            )))
        self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        self.c_ph = tf.placeholder(shape=(None,), dtype=tf.string)
        self.c_emb = self.embed(self.c_ph)
        self.sentencize_fn = sentencize_fn
        # self.sentence_pad_size = 15

    def __call__(self, contexts: List[str]):
        """
        Rank sentences and return top n sentences.
        """

        all_vectors = []
        # contexts_len = []

        for context in contexts:
            # DEBUG
            # start_time = time.time()
            sentences = self.sentencize_fn(context)
            s_vectors = self.session.run([self.c_emb], feed_dict={self.c_ph: sentences})[0]
            # if len(s_vectors[0]) >= self.sentence_pad_size:
            #     m = s_vectors[0][:self.sentence_pad_size]
            #     context_len = len(s_vectors)
            # else:
            #     context_len = len(s_vectors[0])
            #     pad_len = self.sentence_pad_size - context_len
            #     m = np.pad(s_vectors[0], ((0, pad_len), (0, 0)),
            #                mode='constant', constant_values=0.0)
            # all_vectors.append(m)
            # contexts_len.append(context_len)
            sum_vectors = np.sum(s_vectors, axis=0) / s_vectors.shape[0]
            all_vectors.append(sum_vectors)
        return all_vectors
