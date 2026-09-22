"""Tests for Jupyter Notebook (.ipynb) parser and chunker."""

import json
from pathlib import Path
import pytest

from repomind.parsing.notebook_parser import parse_notebook, NotebookParser
from repomind.parsing.models import ParsedFile
from repomind.chunking.models import CodeChunk


def test_normal_code_cells(tmp_path):
    """Test parsing standard code cells without functions/classes."""
    nb_content = json.dumps({
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": ["x = 10\n", "y = 20\n", "print(x + y)"]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    })
    filepath = Path("analysis.ipynb")
    parsed_file, chunks = parse_notebook(nb_content, filepath)

    assert not parsed_file.has_syntax_error
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "notebook_code"
    assert chunks[0].chunk_id == "analysis.ipynb::cell_0::code"
    assert "print(x + y)" in chunks[0].source_code


def test_functions_in_code_cells(tmp_path):
    """Test extracting function info and function chunks from notebook code cells."""
    nb_content = json.dumps({
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "def add(a, b):\n",
                    "    \"\"\"Add two numbers.\"\"\"\n",
                    "    return a + b\n"
                ]
            }
        ],
        "metadata": {}
    })
    filepath = Path("notebook.ipynb")
    parsed_file, chunks = parse_notebook(nb_content, filepath)

    assert len(parsed_file.functions) == 1
    func = parsed_file.functions[0]
    assert func.name == "add"
    assert func.docstring == "Add two numbers."

    func_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(func_chunks) == 1
    assert func_chunks[0].symbol_name == "add"
    assert func_chunks[0].chunk_id == "notebook.ipynb::cell_0::add@1"


def test_classes_in_code_cells(tmp_path):
    """Test extracting class info and class chunks from notebook code cells."""
    nb_content = json.dumps({
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "class Calculator:\n",
                    "    def compute(self, x):\n",
                    "        return x * 2\n"
                ]
            }
        ],
        "metadata": {}
    })
    filepath = Path("calc.ipynb")
    parsed_file, chunks = parse_notebook(nb_content, filepath)

    assert len(parsed_file.classes) == 1
    cls = parsed_file.classes[0]
    assert cls.name == "Calculator"
    assert len(cls.methods) == 1
    assert cls.methods[0].name == "compute"

    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) == 1
    assert class_chunks[0].symbol_name == "Calculator"
    assert class_chunks[0].chunk_id == "calc.ipynb::cell_0::Calculator@1"


def test_same_function_name_in_different_cells(tmp_path):
    """Test that functions with same name in different cells produce distinct cell-scoped chunk IDs."""
    nb_content = json.dumps({
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "def train_model(data):\n",
                    "    return 'v1'\n"
                ]
            },
            {
                "cell_type": "code",
                "source": [
                    "def train_model(data):\n",
                    "    return 'v2'\n"
                ]
            }
        ],
        "metadata": {}
    })
    filepath = Path("experiments.ipynb")
    parsed_file, chunks = parse_notebook(nb_content, filepath)

    func_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(func_chunks) == 2
    assert func_chunks[0].chunk_id == "experiments.ipynb::cell_0::train_model@1"
    assert func_chunks[1].chunk_id == "experiments.ipynb::cell_1::train_model@3"
    assert func_chunks[0].chunk_id != func_chunks[1].chunk_id


def test_cell_syntax_error_fallback(tmp_path):
    """Test that a syntax error in one cell falls back to notebook_code while other cells parse normally."""
    nb_content = json.dumps({
        "cells": [
            {
                "cell_type": "code",
                "source": ["def valid_func():\n", "    return True\n"]
            },
            {
                "cell_type": "code",
                "source": ["def broken_func(:\n", "    return False\n"]
            }
        ],
        "metadata": {}
    })
    filepath = Path("syntax.ipynb")
    parsed_file, chunks = parse_notebook(nb_content, filepath)

    assert not parsed_file.has_syntax_error
    assert len(parsed_file.functions) == 1
    assert parsed_file.functions[0].name == "valid_func"

    assert len(chunks) == 2
    assert chunks[0].chunk_type == "function"
    assert chunks[0].chunk_id == "syntax.ipynb::cell_0::valid_func@1"
    assert chunks[1].chunk_type == "notebook_code"
    assert chunks[1].chunk_id == "syntax.ipynb::cell_1::code"


def test_outputs_and_execution_counts_ignored(tmp_path):
    """Test that cell outputs, execution counts, and attachments are ignored in chunks."""
    nb_content = json.dumps({
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 42,
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {
                            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                            "text/plain": "<Figure size 640x480 with 1 Axes>"
                        }
                    }
                ],
                "source": ["x = 100\n"]
            }
        ],
        "metadata": {}
    })
    filepath = Path("plot.ipynb")
    parsed_file, chunks = parse_notebook(nb_content, filepath)

    assert len(chunks) == 1
    assert "iVBORw0KGgoAAAANSUhEUg" not in chunks[0].source_code
    assert chunks[0].source_code.strip() == "x = 100"


def test_imports_and_calls(tmp_path):
    """Test extracting imports and function calls across notebook cells."""
    nb_content = json.dumps({
        "cells": [
            {
                "cell_type": "code",
                "source": ["import os\n", "from pathlib import Path\n"]
            },
            {
                "cell_type": "code",
                "source": ["def run_pipeline():\n", "    p = Path('file.txt')\n", "    return os.getcwd()\n"]
            }
        ],
        "metadata": {}
    })
    filepath = Path("pipe.ipynb")
    parsed_file, chunks = parse_notebook(nb_content, filepath)

    assert len(parsed_file.imports) == 2
    assert parsed_file.imports[0].module == "os"
    assert parsed_file.imports[1].module == "pathlib"

    callee_names = [c.callee_name for c in parsed_file.calls]
    assert "Path" in callee_names or "getcwd" in callee_names


def test_markdown_cells(tmp_path):
    """Test that markdown cells emit notebook_markdown chunks and set module docstring."""
    nb_content = json.dumps({
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# Data Analysis Notebook\n", "This notebook analyzes sales."]
            },
            {
                "cell_type": "code",
                "source": ["import pandas as pd"]
            }
        ],
        "metadata": {}
    })
    filepath = Path("doc.ipynb")
    parsed_file, chunks = parse_notebook(nb_content, filepath)

    assert parsed_file.module_docstring == "# Data Analysis Notebook\nThis notebook analyzes sales."

    md_chunks = [c for c in chunks if c.chunk_type == "notebook_markdown"]
    assert len(md_chunks) == 1
    assert md_chunks[0].chunk_id == "doc.ipynb::cell_0::markdown"
    assert "# Data Analysis Notebook" in md_chunks[0].source_code


def test_ipython_magic_sanitization(tmp_path):
    """Test that IPython magics and shell commands are sanitized without breaking AST parsing."""
    nb_content = json.dumps({
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "%matplotlib inline\n",
                    "!pip install numpy\n",
                    "import numpy as np\n",
                    "def get_arr():\n",
                    "    return np.array([1, 2, 3])\n",
                    "get_arr()?\n"
                ]
            }
        ],
        "metadata": {}
    })
    filepath = Path("magic.ipynb")
    parsed_file, chunks = parse_notebook(nb_content, filepath)

    assert not parsed_file.has_syntax_error
    assert len(parsed_file.functions) == 1
    assert parsed_file.functions[0].name == "get_arr"


def test_empty_notebook(tmp_path):
    """Test parsing an empty notebook yielding a single file chunk."""
    nb_content = json.dumps({
        "cells": [],
        "metadata": {}
    })
    filepath = Path("empty.ipynb")
    parsed_file, chunks = parse_notebook(nb_content, filepath)

    assert not parsed_file.has_syntax_error
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "file"
    assert chunks[0].chunk_id == "empty.ipynb::file"


def test_markdown_only_notebook(tmp_path):
    """Test notebook with only markdown cells."""
    nb_content = json.dumps({
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# README\n", "Just markdown."]
            }
        ],
        "metadata": {}
    })
    filepath = Path("readme.ipynb")
    parsed_file, chunks = parse_notebook(nb_content, filepath)

    assert not parsed_file.has_syntax_error
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "notebook_markdown"


def test_malformed_json(tmp_path):
    """Test graceful handling of malformed JSON."""
    nb_content = "NOT A VALID JSON {"
    filepath = Path("broken.ipynb")
    parsed_file, chunks = parse_notebook(nb_content, filepath)

    assert parsed_file.has_syntax_error
    assert "Invalid notebook JSON" in parsed_file.syntax_error_message
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "file"


def test_non_python_kernel(tmp_path):
    """Test notebook targeting R kernel emits raw notebook_code chunks without AST parsing."""
    nb_content = json.dumps({
        "cells": [
            {
                "cell_type": "code",
                "source": ["x <- c(1, 2, 3)\n", "plot(x)"]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "R", "language": "R", "name": "ir"}
        }
    })
    filepath = Path("r_analysis.ipynb")
    parsed_file, chunks = parse_notebook(nb_content, filepath)

    assert not parsed_file.has_syntax_error
    assert len(parsed_file.functions) == 0
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "notebook_code"
    assert "x <- c(1, 2, 3)" in chunks[0].source_code
