from typing import List, Union, Any
from operator import itemgetter
import copy
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component
from .utils import flatten_nested_list

logger = get_logger(__name__)


@register("ensemble_ranker")
class EnsembleRanker(Component):

    def __init__(self, top_n=10, active=True, *args, **kwargs):
        self.top_n = top_n
        self.active = active

    def __call__(self, tfidf: List[List[List[Union[Any]]]] = None,
                 tfhub: List[List[List[Union[Any]]]] = None,
                 rnet: List[List[List[Union[Any]]]] = None, *args, **kwargs) -> \
            List[List[List[Union[str, int, float]]]]:

        CHUNK_IDX = 3
        SCORE_IDX = 2
        FAKE_SCORE = 0.001
        NORM_THRESH = 50  # take only the first 50 results to count np.linalg.norm?

        if tfidf is not None:
            tfidf = [[list(el) for el in instance] for instance in tfidf]
        if rnet is not None:
            rnet = [[list(el) for el in instance] for instance in rnet]
        if tfhub is not None:
            tfhub = [[list(el) for el in instance] for instance in tfhub]

        rankers = [r for r in [tfidf, tfhub, rnet] if r is not None]
        num_rankers = len(rankers)

        def update_all_predictions(predictions, ranker_instance):
            for predicted_chunk in ranker_instance:
                chunk_idx = predicted_chunk[CHUNK_IDX]
                if chunk_idx in instance_data_ids:
                    data_idx = list(map(itemgetter(CHUNK_IDX), predictions)).index(chunk_idx)
                    predictions[data_idx][SCORE_IDX] = flatten_nested_list(
                        predictions[data_idx][SCORE_IDX] + [predicted_chunk[SCORE_IDX]])
                else:
                    predicted_chunk[SCORE_IDX] = [predicted_chunk[SCORE_IDX]]
                    predictions.append(predicted_chunk)

        def normalize_scores(ranker_results):
            """
            Normalize paragraph scores with np.linalg.norm
            """
            for instance in ranker_results:
                scores = list(map(itemgetter(SCORE_IDX), instance))
                norm = np.linalg.norm(scores[:NORM_THRESH])
                for pred in instance:
                    pred[SCORE_IDX] = float(pred[SCORE_IDX] / norm)

        # Normalize scores from all tfidf and rnet:
        if tfidf is not None:
            normalize_scores(tfidf)
        if rnet is not None:
            normalize_scores(rnet)

        # Optional
        # if tfhub is not None:
        #     normalize_scores(tfhub)

        # Count average scores from all rankers
        all_data = []
        for instances in zip(*rankers):

            for item in instances[0]:
                item[SCORE_IDX] = [item[SCORE_IDX]]

            instance_predictions = copy.deepcopy(instances[0])
            instance_data_ids = set(map(itemgetter(CHUNK_IDX), instance_predictions))

            for i in range(1, len(instances)):
                update_all_predictions(instance_predictions, instances[i])

            for prediction in instance_predictions:
                len_scores = len(prediction[SCORE_IDX])
                assert len_scores <= num_rankers
                if len_scores < num_rankers:
                    prediction[SCORE_IDX] = np.mean(
                        prediction[SCORE_IDX] + (num_rankers - len_scores) * [FAKE_SCORE])
                else:
                    prediction[SCORE_IDX] = np.mean(prediction[SCORE_IDX])

            instance_predictions = sorted(instance_predictions, key=itemgetter(SCORE_IDX), reverse=True)

            if self.active:
                instance_predictions = instance_predictions[:self.top_n]

            for i in range(len(instance_predictions)):
                instance_predictions[i][0] = i

            all_data.append(instance_predictions)

        return all_data
