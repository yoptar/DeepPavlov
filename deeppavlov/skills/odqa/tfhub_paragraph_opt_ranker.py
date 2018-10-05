"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component
from deeppavlov.skills.odqa.encoders.use_encoder import USEEncoder
from deeppavlov.core.commands.utils import expand_path

logger = get_logger(__name__)


@register("tfhub_paragraph_opt_ranker")
class TFHUBParagraphOptRanker(Component):

    def __init__(self, load_path, encoder: USEEncoder, top_n=10, active: bool = True, **kwargs):
        """
        :param load_path: a path to .npy matrix with context vectors
        :param ranker: an inner ranker, can be USE or ELMo models from tfhub
        :param top_n: top n of document ids to return
        :param active: when is not active, return all doc ids.
        """

        self.load_path = expand_path(load_path)
        self.encoder = encoder
        self.top_n = top_n
        self.active = active
        self.context_vectors = np.load(self.load_path)

    def __call__(self, queries: List[str]):

        q_vectors = self.encoder(queries)
        dots = np.matmul(q_vectors, self.context_vectors.transpose())
        indices = dots.argsort()
        if self.active:
            ids = indices[:, -self.top_n:][::-1]
        else:
            ids = indices[:, ::-1]
        dot_values = np.take(dots, indices)

        # for chainer's sake:
        dot_values = dot_values.tolist()
        ids = ids.tolist()

        return dot_values, ids


# ranker = TFHUBParagraphOptRanker("/media/olga/Data/projects/DeepPavlov/download/general_electrics/use_vectors.npy",
#                                  USEEncoder())
# questions = ["How can I get some?", "What is oil?"]
# res = ranker(questions)
# print(res)