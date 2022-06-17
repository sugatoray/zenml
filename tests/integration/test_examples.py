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
import logging
import os
import platform
import shutil
import time
from abc import ABC
from pathlib import Path
from typing import Callable, List, Optional

import pytest
from pydantic import BaseModel

from zenml.cli import set_active_stack
# from zenml.integrations.slack.alerters import SlackAlerter
from zenml.integrations.mlflow.experiment_trackers import \
    MLFlowExperimentTracker
from zenml.pipelines.run_pipeline import run_pipeline
from zenml.repository import Repository
from zenml.stack import Stack, StackComponent

from .example_validations import generate_basic_validation_function, \
    mlflow_tracking_example_validation

MLFLOW_TRACKING_URI = 'https://a5e1145ed6528439e978f0ead9686b91-' \
                      '1439350632.us-east-1.elb.amazonaws.com'

def copy_example_files(example_dir: str, dst_dir: str) -> None:
    for item in os.listdir(example_dir):
        if item == ".zen":
            # don't copy any existing ZenML repository
            continue

        s = os.path.join(example_dir, item)
        d = os.path.join(dst_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)


class ExampleConfiguration(BaseModel, ABC):
    """Configuration options for testing a ZenML example.

    Attributes:
        name: The name (=directory name) of the example
    """

    name: str
    runs_on_windows: bool
    required_stack_components: List[StackComponent] = list()
    pipeline_name: str
    pipeline_path: str
    step_count: Optional[int]
    validation_function: Optional[Callable] = None

    def run_example(self):
        run_pipeline(python_file=self.pipeline_path, config_path="config.yaml")

    def duplicate_and_update_stack(self) -> None:
        repo = Repository()
        components = repo.active_stack.components

        for component in self.required_stack_components:
            components[component.TYPE] = component
        stack = Stack.from_components(
            name=f"{self.name}_stack", components=components
        )
        repo.register_stack(stack)
        repo.activate_stack(stack.name)

    def assert_successful(self, repo: Repository):
        if self.validation_function:
            return self.validation_function(repo)
        else:
            return generate_basic_validation_function(
                pipeline_name=self.pipeline_name, step_count=self.step_count
            )(repo)


EXAMPLES = [
    ExampleConfiguration(
        name="mlflow_tracking",
        pipeline_path="pipelines/training_pipeline/training_pipeline.py",
        pipeline_name="mlflow_example_pipeline",
        runs_on_windows=True,
        required_stack_components=[
            MLFlowExperimentTracker(
                name="mlflow_tracker",
                tracking_uri=MLFLOW_TRACKING_URI,
                tracking_username='testusr',
                tracking_password='testpwd')
        ],
        validation_function=mlflow_tracking_example_validation)
    # ExampleConfiguration(
    #     name="slack_alert",
    #     pipeline_path="pipelines/post_pipeline.py",
    #     pipeline_name="post_pipeline",
    #     runs_on_windows=True,
    #     required_stack_components=[
    #         SlackAlerter(slack_token="", default_slack_channel_id="")
    #     ],
    #     step_count=5),
    # ExampleConfiguration(
    #     name="lightgbm",
    #     pipeline_path="pipelines/lgbm_pipeline/lgbm_pipeline.py",
    #     pipeline_name="lgbm_pipeline",
    #     runs_on_windows=False,
    #     step_count=3,
    # ),
    # ExampleConfiguration(
    #     name="neural_prophet",
    #     pipeline_path="pipelines/neural_prophet_pipeline/"
    #                   "neural_prophet_pipeline.py",
    #     pipeline_name="neural_prophet_pipeline",
    #     runs_on_windows=False,
    #     step_count=3,
    # ),
    # ExampleConfiguration(
    #     name="scipy",
    #     pipeline_path="pipelines/scipy_example_pipeline/"
    #                   "scipy_example_pipeline.py",
    #     pipeline_name="scipy_example_pipeline",
    #     runs_on_windows=True,
    #     step_count=4,
    # ),
    # ExampleConfiguration(
    #     name="xgboost",
    #     pipeline_path="pipelines/xgboost_pipeline/xgboost_pipeline.py",
    #     pipeline_name="xgboost_pipeline",
    #     runs_on_windows=False,
    #     step_count=3,
    # ),
]


@pytest.mark.parametrize(
    "example_configuration",
    [pytest.param(example, id=example.name) for example in EXAMPLES],
)
def test_run_example(
    example_configuration: ExampleConfiguration,
    tmp_path_factory: pytest.TempPathFactory,
    repo_fixture_name: str,
    request: pytest.FixtureRequest,
    virtualenv: str,
) -> None:
    """Runs the given examples and validates they ran correctly.

    Args:
        example_configuration: Configuration of the example to run.
        tmp_path_factory: Factory to generate temporary test paths.
        repo_fixture_name: Name of a fixture that returns a ZenML repository.
            This fixture will be executed and the example will run on the
            active stack of the repository given by the fixture.
        request: Pytest fixture needed to run the fixture given in the
            `repo_fixture_name` argument
        virtualenv: Either a separate cloned environment for each test, or an
                    empty string.
    """
    if (
        not example_configuration.runs_on_windows
        and platform.system() == "Windows"
    ):
        logging.info(
            f"Skipping example {example_configuration.name} on windows."
        )
        return

    # run the fixture given by repo_fixture_name
    repo = request.getfixturevalue(repo_fixture_name)

    tmp_path = tmp_path_factory.mktemp("tmp")

    # Root directory of all checked out examples
    examples_directory = Path(repo.original_cwd) / "examples"

    # Copy all example files into the repository directory
    copy_example_files(
        str(examples_directory / example_configuration.name), str(tmp_path)
    )

    previous_wd = os.getcwd()
    os.chdir(tmp_path)

    active_stack = Repository().active_stack.name
    Repository.initialize(root=tmp_path)
    Repository._reset_instance()
    set_active_stack(stack_name=active_stack)

    # allow any additional setup that the example might need
    if example_configuration.required_stack_components:
        example_configuration.duplicate_and_update_stack()

    example_configuration.run_example()

    time.sleep(1)
    # Validate the result
    # example_configuration.assert_successful(repo)

    # clean up
    try:
        os.chdir(previous_wd)
        shutil.rmtree(tmp_path)
    except PermissionError:
        # Windows does not have the concept of unlinking a file and deleting
        # once all processes that are accessing the resource are done
        # instead windows tries to delete immediately and fails with a
        # PermissionError: [WinError 32] The process cannot access the
        # file because it is being used by another process
        logging.debug(
            "Skipping deletion of temp dir at teardown, due to "
            "Windows Permission error"
        )
