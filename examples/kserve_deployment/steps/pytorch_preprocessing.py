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

from zenml.steps import step


@step
def build_pytorch_preprocessor() -> torch.nn.Module:
    """Build some pytorch data preprocessing module.

    The resulting transofrmation should either be called within the trainer
    step, or it can be embedded into the dataloader.
    """
    tansformation = torchvision.transforms.Normalize((0.1307,), (0.3081,))
    return tansformation
