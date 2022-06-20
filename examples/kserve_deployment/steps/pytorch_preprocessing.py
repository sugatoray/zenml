#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.


import torch
import torchvision


def normalize_pytorch_tensor(data: torch.Tensor) -> torch.Tensor:
    """Perform some pytorch data preprocessing.

    There are two places where preprocessing can be performed with pytorch:
    - passed in as transformation to the dataloader to load on the fly
    - called within the trainer step after loading the data

    In this example, we call the preprocessing function within the trainer step
    and additionally pass the module path to the custom deployment class.
    """
    tansformation = torchvision.transforms.Normalize((0.1307,), (0.3081,))
    return tansformation(data)
