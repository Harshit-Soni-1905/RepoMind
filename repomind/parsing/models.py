"""Data models for parsed Python code elements.

This module defines immutable data structures representing functions, classes,
imports, and fully parsed files extracted from Python AST.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict


@dataclass(frozen=True)
class FunctionInfo:
    """Represents a parsed function or method.

    Attributes:
        name: Function name
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (inclusive)
        docstring: Function docstring, None if absent
        parameters: List of parameter names (without type annotations)
        decorators: List of decorator names (e.g., ['@staticmethod', '@property'])
        is_async: True for async functions
        is_method: True if this function is inside a class
        parent_class: Name of containing class if is_method is True, None otherwise
    """
    name: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    parameters: List[str] = None
    decorators: List[str] = None
    is_async: bool = False
    is_method: bool = False
    parent_class: Optional[str] = None

    def __post_init__(self):
        # Provide defaults for mutable fields
        if self.parameters is None:
            object.__setattr__(self, 'parameters', [])
        if self.decorators is None:
            object.__setattr__(self, 'decorators', [])


@dataclass(frozen=True)
class ClassInfo:
    """Represents a parsed class definition.

    Attributes:
        name: Class name
        start_line: Starting line number
        end_line: Ending line number
        docstring: Class docstring, None if absent
        bases: List of base class names
        decorators: List of decorator names
        methods: List of FunctionInfo objects for methods in this class
    """
    name: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    bases: List[str] = None
    decorators: List[str] = None
    methods: List[FunctionInfo] = None

    def __post_init__(self):
        # Provide defaults for mutable fields
        if self.bases is None:
            object.__setattr__(self, 'bases', [])
        if self.decorators is None:
            object.__setattr__(self, 'decorators', [])
        if self.methods is None:
            object.__setattr__(self, 'methods', [])


@dataclass(frozen=True)
class ImportInfo:
    """Represents an import statement.

    Attributes:
        module: Module name being imported (e.g., 'os', 'os.path', 'mypackage')
        names: For 'from X import a, b' → ['a', 'b']. Empty for 'import X'.
        aliases: Mapping of imported name to alias (e.g., {'numpy': 'np'})
        line: Line number where import appears
        is_from_import: True for 'from X import Y', False for 'import X'
    """
    module: str
    line: int
    is_from_import: bool = False
    names: List[str] = None
    aliases: Dict[str, str] = None

    def __post_init__(self):
        # Provide defaults for mutable fields
        if self.names is None:
            object.__setattr__(self, 'names', [])
        if self.aliases is None:
            object.__setattr__(self, 'aliases', {})


@dataclass(frozen=True)
class CallInfo:
    """Represents a function/method call site.

    Attributes:
        caller_name: Name of the calling function/method (None if at module level)
        caller_type: Type of caller: "function", "method", or "module"
        caller_class: Class name if caller is a method, None otherwise
        caller_start_line: Start line of caller function/method
        callee_name: Name of the called function/method
        callee_qualname: Qualified name if resolvable (e.g., "module.func", "Class.method")
        call_line: Line number where the call occurs
        is_resolved: True if callee could be statically resolved to a known symbol
        resolution_type: How the call was resolved ("local", "imported", "method", "unresolved")
    """
    caller_name: Optional[str]
    caller_type: str  # "function", "method", "module"
    caller_class: Optional[str]
    caller_start_line: int
    callee_name: str
    callee_qualname: Optional[str]
    call_line: int
    is_resolved: bool
    resolution_type: str


@dataclass(frozen=True)
class ParsedFile:
    """Represents a fully parsed Python source file.

    Attributes:
        filepath: Path relative to repository root
        functions: List of top-level functions in the file
        classes: List of top-level classes in the file
        imports: List of all import statements
        calls: List of function/method call sites
        module_docstring: Module-level docstring (first string literal), None if absent
        has_syntax_error: True if file could not be parsed due to syntax error
        syntax_error_message: Error message if has_syntax_error is True, None otherwise
    """
    filepath: Path
    functions: List[FunctionInfo] = None
    classes: List[ClassInfo] = None
    imports: List[ImportInfo] = None
    calls: List[CallInfo] = None
    module_docstring: Optional[str] = None
    has_syntax_error: bool = False
    syntax_error_message: Optional[str] = None

    def __post_init__(self):
        # Provide defaults for mutable fields
        if self.functions is None:
            object.__setattr__(self, 'functions', [])
        if self.classes is None:
            object.__setattr__(self, 'classes', [])
        if self.imports is None:
            object.__setattr__(self, 'imports', [])
        if self.calls is None:
            object.__setattr__(self, 'calls', [])
