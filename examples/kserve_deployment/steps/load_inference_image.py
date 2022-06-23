#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import base64
import json

import requests

from zenml.steps import step
from zenml.steps.base_step_config import BaseStepConfig


class LoadInferenceImageStepConfig(BaseStepConfig):
    """
    Configuration for the LoadInferenceImageStepConfig.
    """

    img_url: str = "https://github.com/kserve/kserve/blob/master/docs/samples/v1beta1/torchserve/v1/imgconv/1.png"


@step(enable_cache=False)
def load_inference_image(config: LoadInferenceImageStepConfig) -> str:
    """
    Loads an image from a URL and returns the image as a base64 encoded string.

    Args:
        config: Configuration for the ImageImporterStep.

    Returns:
        Output(data=Dict[str, List]): The image as a base64 encoded string.
    """
    img_data = requests.get(config.img_url).content
    image_64_encode = base64.b64encode(img_data)
    bytes_array = image_64_encode.decode("utf-8")
    request = {"data": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAw0lEQVR4nGNgGFggVVj4/y8Q2GOR83n+58/fP0DwcSqmpNN7oOTJw6f+/H2pjUU2JCSEk0EWqN0cl828e/FIxvz9/9cCh1zS5z9/G9mwyzl/+PNnKQ45nyNAr9ThMHQ/UG4tDofuB4bQIhz6fIBenMWJQ+7Vn7+zeLCbKXv6z59NOPQVgsIcW4QA9YFi6wNQLrKwsBebW/68DJ388Nun5XFocrqvIFH59+XhBAxThTfeB0r+vP/QHbuDCgr2JmOXoSsAAKK7bU3vISS4AAAAAElFTkSuQmCC"}
    return json.dumps([request])
