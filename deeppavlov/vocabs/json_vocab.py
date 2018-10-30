from typing import List, Any, Union

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.utils import expand_path

logger = get_logger(__name__)


@register('json_vocab')
class JSONVocab:
    """
    Get SQlite documents by ids.
    """

    def __init__(self, load_path, return_all_content=False, *args, **kwargs):
        """
        Read json data and index it. Get docs by indices.
        """
        self.json = read_json(expand_path(load_path))
        self.index2doc = self._get_index2doc()
        self.return_all_content = return_all_content

    def _get_index2doc(self):
        return {i: doc for i, doc in enumerate(self.json)}

    def __call__(self, doc_ids: List[List[Any]], *args, **kwargs) -> Union[List[List[str]], List[Any]]:
        batch_docs = []
        if self.return_all_content:
            doc_ids = [self.index2doc.keys()]
        for instance_ids in doc_ids:
            batch_docs.append([self.index2doc[i] for i in instance_ids])
        return batch_docs
