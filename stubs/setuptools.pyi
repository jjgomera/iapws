#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""IAPWS stubs for setuptools"""

from typing import Any, List

def setup(name: str,
          version: Any,
          packages: List[str],
          include_package_data: bool,
          author: str,
          author_email: str,
          url: str,
          download_url: str,
          description: str,
          long_description: str,
          license: str,
          python_requires: str,
          install_requires: List[str],
          classifiers: List[str]
) -> None:
    ...
