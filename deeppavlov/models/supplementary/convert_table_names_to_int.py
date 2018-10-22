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
from typing import List, Any, Union
from operator import itemgetter

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register


@register("table_names_converter")
class TableNamesConverter(Component):

    def __init__(self, converted_type: str = 'list', *args, **kwargs):
        """
        Format table names.
        Args:
            converted_type: a string with converted type. Can be from {'list', 'ensemble'}
            *args:
            **kwargs:
        """
        self.converted_type = converted_type

    def __call__(self, batch_data: Any, *args, **kwargs):
        if self.converted_type == 'list':
            return self._convert_list(batch_data)
        elif self.converted_type == 'ensemble':
            converted = self._convert_ensemble(batch_data)
            if isinstance(converted[0][0], str):
                return self._convert_list(converted)
            return converted
        else:
            raise RuntimeError(f'No such conversion option in {self.__class__.__name__}')

    @staticmethod
    def _convert_list(batch_titles: List[List[str]]) -> List[List[int]]:
        all_titles = []
        for titles in batch_titles:
            all_titles.append([int(title.split('.')[0]) for title in titles])
        return all_titles

    @staticmethod
    def _convert_ensemble(batch_data: List[List[List[Union[str, int, float]]]]) -> List[List[str]]:
        title_index = 3
        all_titles = []
        for data in batch_data:
            # instance_data = []
            # for i in data:
            #     instance_data.append(list(map(itemgetter(title_index), i)))
            all_titles.append(list(map(itemgetter(title_index), data)))
        return all_titles


@register("table_indices_converter")
class TableIndicesConverter(Component):

    def __init__(self, suffix='.txt', *args, **kwargs):
        self.suffix = suffix

    def __call__(self, batch_data: List[List[int]], *args, **kwargs):
        res = []
        for instance_data in batch_data:
            res.append([f'{i}{self.suffix}' for i in instance_data])
        return res
