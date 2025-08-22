# Copyright 2024 Google LLC
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

import os
import re
import subprocess
import sys
from distutils.cmd import Command
from pathlib import Path

from setuptools import Extension, setup, find_packages

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="splinetracer",
    version="0.1.1",
    author="Alexander Mai",
    author_email="alexandertmai@gmail.com",
    description="Official implementation of our ellipsoid tracing paper",
    long_description="",
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[],
    packages=find_packages(),
    package_data={"build.splinetracer.extension.splinetracer_cpp_extension": ["py.typed", "**/*.pyi"]},
    entry_points={
    },
)

