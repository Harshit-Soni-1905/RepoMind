"""Tests for AST parser."""

import pytest
from pathlib import Path

from repomind.parsing.ast_parser import parse, CodeElementExtractor
from repomind.parsing.models import FunctionInfo, ClassInfo, ImportInfo, ParsedFile


def test_parse_simple_function():
    """Parser should extract a simple function."""
    source = """
def greet(name):
    '''Say hello to someone.'''
    return f"Hello, {name}"
"""
    parsed = parse(source, Path("test.py"))

    assert not parsed.has_syntax_error
    assert len(parsed.functions) == 1

    func = parsed.functions[0]
    assert func.name == "greet"
    assert func.docstring == "Say hello to someone."
    assert func.parameters == ["name"]
    assert func.is_async is False
    assert func.is_method is False
    assert func.start_line == 2
    assert func.end_line == 4


def test_parse_function_without_docstring():
    """Parser should handle functions without docstrings."""
    source = "def foo(x, y):\n    return x + y"
    parsed = parse(source, Path("test.py"))

    assert len(parsed.functions) == 1
    func = parsed.functions[0]
    assert func.name == "foo"
    assert func.docstring is None
    assert func.parameters == ["x", "y"]


def test_parse_function_with_decorators():
    """Parser should extract decorator names."""
    source = """
@staticmethod
@cache
def compute():
    return 42
"""
    parsed = parse(source, Path("test.py"))

    assert len(parsed.functions) == 1
    func = parsed.functions[0]
    assert "@staticmethod" in func.decorators
    assert "@cache" in func.decorators


def test_parse_async_function():
    """Parser should recognize async functions."""
    source = """
async def fetch_data():
    return await some_api()
"""
    parsed = parse(source, Path("test.py"))

    assert len(parsed.functions) == 1
    func = parsed.functions[0]
    assert func.name == "fetch_data"
    assert func.is_async is True


def test_parse_simple_class():
    """Parser should extract a simple class."""
    source = """
class Person:
    '''Represents a person.'''
    def __init__(self, name):
        self.name = name
"""
    parsed = parse(source, Path("test.py"))

    assert not parsed.has_syntax_error
    assert len(parsed.classes) == 1

    cls = parsed.classes[0]
    assert cls.name == "Person"
    assert cls.docstring == "Represents a person."
    assert len(cls.methods) == 1
    assert cls.methods[0].name == "__init__"
    assert cls.methods[0].is_method is True
    assert cls.methods[0].parent_class == "Person"


def test_parse_class_with_multiple_methods():
    """Parser should extract all methods from a class."""
    source = """
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b
"""
    parsed = parse(source, Path("test.py"))

    assert len(parsed.classes) == 1
    cls = parsed.classes[0]
    assert cls.name == "Calculator"
    assert len(cls.methods) == 3

    method_names = [m.name for m in cls.methods]
    assert "add" in method_names
    assert "subtract" in method_names
    assert "multiply" in method_names


def test_parse_class_with_inheritance():
    """Parser should extract base classes."""
    source = """
class Child(Parent, Mixin):
    pass
"""
    parsed = parse(source, Path("test.py"))

    assert len(parsed.classes) == 1
    cls = parsed.classes[0]
    assert "Parent" in cls.bases
    assert "Mixin" in cls.bases


def test_parse_class_with_decorators():
    """Parser should extract class decorators."""
    source = """
@dataclass
@frozen
class Point:
    x: int
    y: int
"""
    parsed = parse(source, Path("test.py"))

    assert len(parsed.classes) == 1
    cls = parsed.classes[0]
    assert "@dataclass" in cls.decorators
    assert "@frozen" in cls.decorators


def test_parse_import_statement():
    """Parser should extract import statements."""
    source = """
import os
import sys
"""
    parsed = parse(source, Path("test.py"))

    assert len(parsed.imports) == 2

    modules = [imp.module for imp in parsed.imports]
    assert "os" in modules
    assert "sys" in modules

    for imp in parsed.imports:
        assert imp.is_from_import is False


def test_parse_import_with_alias():
    """Parser should extract import aliases."""
    source = "import numpy as np"
    parsed = parse(source, Path("test.py"))

    assert len(parsed.imports) == 1
    imp = parsed.imports[0]
    assert imp.module == "numpy"
    assert imp.aliases.get("numpy") == "np"


def test_parse_from_import():
    """Parser should extract from-import statements."""
    source = """
from os import path
from typing import List, Dict, Optional
"""
    parsed = parse(source, Path("test.py"))

    assert len(parsed.imports) == 2

    # First import
    os_import = [imp for imp in parsed.imports if imp.module == "os"][0]
    assert os_import.is_from_import is True
    assert "path" in os_import.names

    # Second import
    typing_import = [imp for imp in parsed.imports if imp.module == "typing"][0]
    assert typing_import.is_from_import is True
    assert "List" in typing_import.names
    assert "Dict" in typing_import.names
    assert "Optional" in typing_import.names


def test_parse_from_import_with_alias():
    """Parser should extract from-import with aliases."""
    source = "from collections import defaultdict as dd"
    parsed = parse(source, Path("test.py"))

    assert len(parsed.imports) == 1
    imp = parsed.imports[0]
    assert imp.module == "collections"
    assert "defaultdict" in imp.names
    assert imp.aliases.get("defaultdict") == "dd"


def test_parse_module_docstring():
    """Parser should extract module-level docstring."""
    source = '''
"""This is a module for testing.

It demonstrates module docstring extraction.
"""

def foo():
    pass
'''
    parsed = parse(source, Path("test.py"))

    assert parsed.module_docstring is not None
    assert "This is a module for testing" in parsed.module_docstring


def test_parse_mixed_file():
    """Parser should handle files with functions, classes, and imports."""
    source = """
import os
from typing import List

def utility_function():
    return 42

class MyClass:
    def method(self):
        pass

def another_function():
    pass
"""
    parsed = parse(source, Path("test.py"))

    assert len(parsed.imports) == 2
    assert len(parsed.functions) == 2  # Top-level functions only
    assert len(parsed.classes) == 1
    assert len(parsed.classes[0].methods) == 1


def test_parse_empty_file():
    """Parser should handle empty files gracefully."""
    source = ""
    parsed = parse(source, Path("empty.py"))

    assert not parsed.has_syntax_error
    assert len(parsed.functions) == 0
    assert len(parsed.classes) == 0
    assert len(parsed.imports) == 0
    assert parsed.module_docstring is None


def test_parse_file_with_only_comments():
    """Parser should handle files with only comments."""
    source = """
# This is a comment
# Another comment
"""
    parsed = parse(source, Path("comments.py"))

    assert not parsed.has_syntax_error
    assert len(parsed.functions) == 0
    assert len(parsed.classes) == 0


def test_parse_syntax_error():
    """Parser should gracefully handle syntax errors."""
    source = """
def broken_function(
    # Missing closing parenthesis
    return 42
"""
    parsed = parse(source, Path("broken.py"))

    assert parsed.has_syntax_error
    assert parsed.syntax_error_message is not None
    assert "SyntaxError" in parsed.syntax_error_message
    assert len(parsed.functions) == 0
    assert len(parsed.classes) == 0


def test_parse_incomplete_code():
    """Parser should handle incomplete code."""
    source = "def incomplete("
    parsed = parse(source, Path("incomplete.py"))

    assert parsed.has_syntax_error


def test_parse_nested_functions():
    """Parser should not extract nested functions (they're handled in chunking)."""
    source = """
def outer():
    def inner():
        return 42
    return inner()
"""
    parsed = parse(source, Path("test.py"))

    # Only outer function should be extracted
    assert len(parsed.functions) == 1
    assert parsed.functions[0].name == "outer"


def test_parse_static_and_class_methods():
    """Parser should extract static and class methods."""
    source = """
class MyClass:
    @staticmethod
    def static_method():
        pass

    @classmethod
    def class_method(cls):
        pass

    def instance_method(self):
        pass
"""
    parsed = parse(source, Path("test.py"))

    assert len(parsed.classes) == 1
    cls = parsed.classes[0]
    assert len(cls.methods) == 3

    # Check decorators are captured
    static = [m for m in cls.methods if m.name == "static_method"][0]
    assert "@staticmethod" in static.decorators

    classm = [m for m in cls.methods if m.name == "class_method"][0]
    assert "@classmethod" in classm.decorators


def test_parse_complex_decorators():
    """Parser should handle decorators with arguments."""
    source = """
@decorator_with_args(arg1, arg2)
def decorated_func():
    pass
"""
    parsed = parse(source, Path("test.py"))

    assert len(parsed.functions) == 1
    func = parsed.functions[0]
    # Should extract decorator name even if it has args
    assert len(func.decorators) == 1
    assert "@decorator_with_args" in func.decorators


def test_parse_class_without_methods():
    """Parser should handle classes without methods."""
    source = """
class Empty:
    pass
"""
    parsed = parse(source, Path("test.py"))

    assert len(parsed.classes) == 1
    cls = parsed.classes[0]
    assert cls.name == "Empty"
    assert len(cls.methods) == 0


def test_parse_function_with_no_parameters():
    """Parser should handle functions with no parameters."""
    source = "def no_params():\n    return 42"
    parsed = parse(source, Path("test.py"))

    assert len(parsed.functions) == 1
    func = parsed.functions[0]
    assert func.parameters == []


def test_parse_line_numbers():
    """Parser should accurately capture line numbers."""
    source = """# Line 1
def first():  # Line 2
    pass  # Line 3

class Second:  # Line 5
    def method(self):  # Line 6
        pass  # Line 7
"""
    parsed = parse(source, Path("test.py"))

    func = parsed.functions[0]
    assert func.start_line == 2
    assert func.end_line == 3

    cls = parsed.classes[0]
    assert cls.start_line == 5
    assert cls.end_line == 7


def test_parse_preserves_filepath():
    """Parser should preserve the filepath in ParsedFile."""
    filepath = Path("src/module.py")
    source = "def foo(): pass"
    parsed = parse(source, filepath)

    assert parsed.filepath == filepath


def test_parsed_file_immutability():
    """ParsedFile should be immutable."""
    source = "def foo(): pass"
    parsed = parse(source, Path("test.py"))

    with pytest.raises(AttributeError):
        parsed.functions = []

    with pytest.raises(AttributeError):
        parsed.has_syntax_error = True


def test_function_info_immutability():
    """FunctionInfo should be immutable."""
    func = FunctionInfo(
        name="test",
        start_line=1,
        end_line=2,
        parameters=["x"],
    )

    with pytest.raises(AttributeError):
        func.name = "modified"


def test_class_info_immutability():
    """ClassInfo should be immutable."""
    cls = ClassInfo(
        name="Test",
        start_line=1,
        end_line=5,
        methods=[],
    )

    with pytest.raises(AttributeError):
        cls.name = "Modified"
