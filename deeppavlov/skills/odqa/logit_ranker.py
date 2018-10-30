# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union, Tuple, Any
from operator import itemgetter
from itertools import chain

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Component
from deeppavlov.core.common.chainer import Chainer

logger = get_logger(__name__)


@register("logit_ranker")
class LogitRanker(Component):
    """Select best answer using squad model logits. Make several batches for a single batch, send each batch
     to the squad model separately and get a single best answer for each batch.
     Args:
        squad_model: a loaded squad model
        batch_size: batch size to use with squad model
     Attributes:
        squad_model: a loaded squad model
        batch_size: batch size to use with squad model
    """

    def __init__(self, top_n: 3, squad_model: Union[Chainer, Component], batch_size: int = 50,
                 sort_noans: bool = False, **kwargs):
        self.squad_model = squad_model
        self.batch_size = batch_size
        self.top_n = top_n
        self.sort_noans = sort_noans

    def __call__(self, contexts_batch: List[List[str]], questions_batch: List[List[str]]) -> \
            Tuple[List[List[Tuple[str, Any]]], List[List[int]]]:
        """
        Sort obtained results from squad reader by logits and get the answer with a maximum logit.
        Args:
            contexts_batch: a batch of contexts which should be treated as a single batch in the outer JSON config
            questions_batch: a batch of questions which should be treated as a single batch in the outer JSON config
        Returns:
            a batch of best answers
        """

        batch_best_answers = []
        batch_context_indices = []
        for contexts, questions in zip(contexts_batch, questions_batch):
            results = []
            for i in range(0, len(contexts), self.batch_size):
                c_batch = contexts[i: i + self.batch_size]
                q_batch = questions[i: i + self.batch_size]
                batch_predict = zip(*self.squad_model(c_batch, q_batch))
                results += batch_predict
            if self.sort_noans:
                results = sorted(results, key=lambda x: x[0] == '')
            sorted_items = sorted(enumerate(results), key=lambda x: x[1][2], reverse=True)[:self.top_n]
            best_answers = list(map(itemgetter(1), sorted_items))
            context_indices = list(map(itemgetter(0), sorted_items))
            best_answers = [ba[::2] for ba in best_answers]
            best_answers = [('no answer', ba[1]) if ba[0] == '' else ba for ba in best_answers]
            batch_best_answers.append(best_answers)
            batch_context_indices.append(context_indices)
        return batch_best_answers, batch_context_indices


@register("tables_logit_ranker")
class TablesLogitRanker(Component):
    """Select best answer using squad model logits. Make several batches for a single batch, send each batch
     to the squad model separately and get a single best answer for each batch.
     Args:
        squad_model: a loaded squad model
        batch_size: batch size to use with squad model
     Attributes:
        squad_model: a loaded squad model
        batch_size: batch size to use with squad model
    """

    def __init__(self, top_n: 3, squad_model: Union[Chainer, Component], batch_size: int = 50,
                 sort_noans: bool = False, **kwargs):
        self.squad_model = squad_model
        self.batch_size = batch_size
        self.top_n = top_n
        self.sort_noans = sort_noans

    def __call__(self, contexts_batch: List[List[str]], questions_batch: List[List[str]], cells_batch,
                 batch_table_indices):
        """
        Sort obtained results from squad reader by logits and get the answer with a maximum logit.
        Args:
            contexts_batch: a batch of contexts which should be treated as a single batch in the outer JSON config
            questions_batch: a batch of questions which should be treated as a single batch in the outer JSON config
        Returns:
            a batch of best answers
        """

        batch_best_answers = []
        # batch_context_indices = []
        for instance_contexts, instance_cells in zip(contexts_batch, cells_batch):
            instance_best_answers = []
            # instance_context_indices = []
            for contexts, cells, questions in zip(instance_contexts, instance_cells, questions_batch):
                questions = [questions[0]] * len(contexts)
                results = []
                for i in range(0, len(contexts), self.batch_size):
                    c_batch = contexts[i: i + self.batch_size]
                    q_batch = questions[i: i + self.batch_size]
                    batch_predict = zip(*self.squad_model(c_batch, q_batch))
                    results += batch_predict
                if self.sort_noans:
                    results = sorted(enumerate(results), key=lambda x: x[1][0] == '')
                sorted_items = sorted(results, key=lambda x: x[1][2], reverse=True)
                best_answers = list(map(itemgetter(1), sorted_items))
                context_indices = list(map(itemgetter(0), sorted_items))
                # best_answers = [ba[::2] for ba in best_answers]
                best_answers = [(cells[idx], ba[2]) for idx, ba in zip(context_indices, best_answers)]
                best_answers = [('no answer', ba[1]) if ba[0] == '' else ba for ba in best_answers]
                instance_best_answers.append(best_answers)
                # instance_context_indices.append(context_indices)
            batch_best_answers.append(instance_best_answers)
            # batch_context_indices.append(instance_context_indices)

        # add table indices
        for iba, ti in zip(batch_best_answers, batch_table_indices):
            for k, iiba in enumerate(iba):
                for j in range(len(iiba)):
                    iiba[j] = (iiba[j][0], iiba[j][1], ti[k])
        batch_best_answers = [sorted(chain.from_iterable(iba), key=itemgetter(1), reverse=True)[:self.top_n] for iba in
                              batch_best_answers]

        return batch_best_answers
