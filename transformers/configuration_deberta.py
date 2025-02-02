# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" RoBERTa configuration """

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

from .configuration_bert import BertConfig

logger = logging.getLogger(__name__)

DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    #"microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/config.json",
    "microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-base/resolve/main/config.json",
}


class DebertaConfig(BertConfig):
    pretrained_config_archive_map = DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP