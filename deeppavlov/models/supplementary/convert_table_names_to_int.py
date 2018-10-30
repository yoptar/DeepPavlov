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

    def __init__(self, suffix='.txt', lists=True, *args, **kwargs):
        self.suffix = suffix
        if lists:
            self.title_map = {'0.txt': 0, '1.txt': 1, '10.txt': 2, '11.txt': 3, '12.txt': 4, '13.txt': 5, '14.txt': 6,
                              '15.txt': 7, '16.txt': 8, '17.txt': 9, '18.txt': 10, '19.txt': 11, '2.txt': 12,
                              '20.txt': 13, '21.txt': 14, '22.txt': 15, '23.txt': 16, '24.txt': 17, '25.txt': 18,
                              '26.txt': 19, '27.txt': 20, '28.txt': 21, '29.txt': 22, '3.txt': 23, '30.txt': 24,
                              '31.txt': 25, '32.txt': 26, '33.txt': 27, '34.txt': 28, '35.txt': 29, '36.txt': 30,
                              '37.txt': 31, '38.txt': 32, '39.txt': 33, '4.txt': 34, '40.txt': 35, '41.txt': 36,
                              '42.txt': 37, '43.txt': 38, '44.txt': 39, '45.txt': 40, '5.txt': 41, '6.txt': 42,
                              '7.txt': 43, '8.txt': 44, '9.txt': 45}
        else:
            self.title_map = {'0.txt': 0, '1.txt': 1, '10.txt': 2, '11.txt': 3, '12.txt': 4, '13.txt': 5, '14.txt': 6,
                              '15.txt': 7, '16.txt': 8, '17.txt': 9, '2.txt': 10, '3.txt': 11, '4.txt': 12, '5.txt': 13,
                              '6.txt': 14, '7.txt': 15, '8.txt': 16, '9.txt': 17}

    def __call__(self, batch_data: List[List[int]], *args, **kwargs):
        res = []
        for instance_data in batch_data:
            # res.append([k for i in instance_data for k in self.title_map.keys() if self.title_map[k] == i])
            _res = []
            for i in instance_data:
                if i in self.title_map.keys():
                    _res.append(self.title_map[i])
            res.append(_res)

        return res
