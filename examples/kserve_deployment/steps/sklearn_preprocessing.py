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

import numpy as np
from sklearn.preprocessing import StandardScaler

from zenml.steps import Output, step


@step
def fit_standard_scaler(
    data: np.ndarray,
) -> Output(scaler=StandardScaler, transformed_data=np.ndarray):
    """Fit a sklearn StandardScaler on the given 2D numeric numpy data."""
    scaler = StandardScaler()
    transformed_data = scaler.fit_transform(data)
    return scaler, transformed_data


@step
def apply_standard_scaler(
    scaler: StandardScaler, data: np.ndarray
) -> np.ndarray:
    """Standardize 2D numeric NumPy data using a prefitted StandardScaler."""
    return scaler.transform(data)
