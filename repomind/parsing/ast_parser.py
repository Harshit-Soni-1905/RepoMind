"""Python AST parser for extracting code elements.

This module uses Python's built-in ast module to parse source code and extract
functions, classes, imports, and function/method calls with their metadata.
"""

import ast
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple

from repomind.parsing.models import (
    FunctionInfo,
    ClassInfo,
    ImportInfo,
    CallInfo,
    ParsedFile,
)


class CodeElementExtractor(ast.NodeVisitor):
    """AST visitor that extracts functions, classes, imports, and calls.

    Uses the visitor pattern to walk the AST and collect code elements.
    Tracks current class/function context to distinguish methods from top-level functions
    and to attribute calls to their callers.
    """

    def __init__(self):
        self.functions: list[FunctionInfo] = []
        self.classes: list[ClassInfo] = []
        self.imports: list[ImportInfo] = []
        self.calls: list[CallInfo] = []
        self.module_docstring: Optional[str] = None
        self.current_class: Optional[str] = None
        self.current_class_methods: list[FunctionInfo] = []
        self.current_function: Optional[FunctionInfo] = None
        self.function_stack: List[FunctionInfo] = []

        # Build import mapping for call resolution
        self._imported_names: Dict[str, str] = {}  # alias -> full module path
        self._imported_modules: Set[str] = set()   # imported module names
        self._from_imports: Dict[str, str] = {}    # imported name -> module
        self._local_functions: Dict[str, FunctionInfo] = {}  # name -> FunctionInfo
        self._local_classes: Dict[str, ClassInfo] = {}       # name -> ClassInfo

    def visit_Module(self, node: ast.Module) -> None:
        """Visit module node and extract module docstring."""
        # Module docstring is the first statement if it's an Expr with a Constant string value
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            self.module_docstring = node.body[0].value.value

        # Visit all statements, but skip the docstring to avoid duplicate processing
        # We need to process imports first for call resolution
        for stmt in node.body:
            if stmt is node.body[0] and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue  # Skip docstring
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition and extract function metadata."""
        func_info = self._extract_function_info(node, is_async=False)

        is_top_level = not self.current_class and not self.function_stack

        if self.current_class:
            # This is a method
            self.current_class_methods.append(func_info)
        elif is_top_level:
            # Top-level function (not nested)
            self.functions.append(func_info)
            self._local_functions[func_info.name] = func_info
        # Nested functions are intentionally not added to functions list
        # (handled during chunking stage)

        # Push onto function stack for tracking caller context
        self.function_stack.append(func_info)
        old_caller = self.current_function
        self.current_function = func_info

        # Visit function body to find calls
        self.generic_visit(node)

        # Restore caller context
        self.function_stack.pop()
        self.current_function = old_caller

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        func_info = self._extract_function_info(node, is_async=True)

        is_top_level = not self.current_class and not self.function_stack

        if self.current_class:
            self.current_class_methods.append(func_info)
        elif is_top_level:
            self.functions.append(func_info)
            self._local_functions[func_info.name] = func_info

        # Push onto function stack for tracking caller context
        self.function_stack.append(func_info)
        old_caller = self.current_function
        self.current_function = func_info

        # Visit function body to find calls
        self.generic_visit(node)

        # Restore caller context
        self.function_stack.pop()
        self.current_function = old_caller

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition and extract class metadata."""
        # Save previous class context (for nested classes)
        old_class = self.current_class
        old_methods = self.current_class_methods

        # Enter new class context
        self.current_class = node.name
        self.current_class_methods = []

        # Extract class info
        class_info = self._extract_class_info(node)

        # Visit class body (this will populate self.current_class_methods)
        self.generic_visit(node)

        # Attach methods to class
        class_info = ClassInfo(
            name=class_info.name,
            start_line=class_info.start_line,
            end_line=class_info.end_line,
            docstring=class_info.docstring,
            bases=class_info.bases,
            decorators=class_info.decorators,
            methods=self.current_class_methods,
        )

        # Add class to list (only top-level classes for now)
        if old_class is None:
            self.classes.append(class_info)

        # Restore previous class context
        self.current_class = old_class
        self.current_class_methods = old_methods

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement (e.g., import os, import sys as system)."""
        for alias in node.names:
            import_info = ImportInfo(
                module=alias.name,
                line=node.lineno,
                is_from_import=False,
                names=[],
                aliases={alias.name: alias.asname} if alias.asname else {},
            )
            self.imports.append(import_info)

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statement (e.g., from os import path)."""
        module = node.module or ""  # 'from . import x' has module=None
        names = [alias.name for alias in node.names]
        aliases = {
            alias.name: alias.asname for alias in node.names if alias.asname
        }

        import_info = ImportInfo(
            module=module,
            line=node.lineno,
            is_from_import=True,
            names=names,
            aliases=aliases,
        )
        self.imports.append(import_info)

        # Track imported names for call resolution
        for alias in node.names:
            imported_name = alias.name
            alias_name = alias.asname or alias.name
            if module:
                self._from_imports[alias_name] = module
            # For relative imports, module might be empty

        self.generic_visit(node)

    def _extract_function_info(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool
    ) -> FunctionInfo:
        """Extract metadata from a function/method node."""
        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract parameter names
        parameters = [arg.arg for arg in node.args.args]

        # Extract decorator names
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]

        # Determine line range
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, "end_lineno") else node.lineno

        return FunctionInfo(
            name=node.name,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring,
            parameters=parameters,
            decorators=decorators,
            is_async=is_async,
            is_method=self.current_class is not None,
            parent_class=self.current_class,
        )

    def _extract_class_info(self, node: ast.ClassDef) -> ClassInfo:
        """Extract metadata from a class node (without methods, added later)."""
        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract base class names
        bases = [self._get_name(base) for base in node.bases]

        # Extract decorator names
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]

        # Determine line range
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, "end_lineno") else node.lineno

        return ClassInfo(
            name=node.name,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring,
            bases=bases,
            decorators=decorators,
            methods=[],  # Will be populated by visit_ClassDef
        )

    def _get_decorator_name(self, node: ast.expr) -> str:
        """Extract decorator name as a string.

        Handles: @decorator, @decorator(), @module.decorator
        """
        if isinstance(node, ast.Name):
            return f"@{node.id}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return f"@{node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                return f"@{self._get_attribute_name(node.func)}"
        elif isinstance(node, ast.Attribute):
            return f"@{self._get_attribute_name(node)}"
        return "@<unknown>"

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get fully qualified attribute name (e.g., module.submodule.attr)."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def _get_name(self, node: ast.expr) -> str:
        """Extract name from a node (for base classes, etc.)."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        return "<unknown>"

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function/method call and extract call information."""
        # Only process calls inside functions/methods (not at module level)
        if not self.current_function:
            self.generic_visit(node)
            return

        caller = self.current_function
        call_line = node.lineno

        # Extract callee name and qualified name
        callee_name, callee_qualname, is_resolved, resolution_type = self._resolve_call(node)

        # Build call info
        call_info = CallInfo(
            caller_name=caller.name,
            caller_type="method" if self.current_class else "function",
            caller_class=self.current_class,
            caller_start_line=caller.start_line,
            callee_name=callee_name,
            callee_qualname=callee_qualname,
            call_line=call_line,
            is_resolved=is_resolved,
            resolution_type=resolution_type,
        )
        self.calls.append(call_info)

        # Continue visiting children
        self.generic_visit(node)

    def _resolve_call(self, node: ast.Call) -> Tuple[str, Optional[str], bool, str]:
        """Resolve a call node to determine callee identity and resolution type.

        Returns:
            Tuple of (callee_name, callee_qualname, is_resolved, resolution_type)
        """
        func = node.func

        # Case 1: Direct name call - func_name()
        if isinstance(func, ast.Name):
            name = func.id
            return self._resolve_name_call(name)

        # Case 2: Attribute call - obj.method() or module.func()
        elif isinstance(func, ast.Attribute):
            return self._resolve_attribute_call(func)

        # Case 3: Other (lambda, etc.) - cannot resolve
        return ("<unknown>", None, False, "unresolved")

    def _resolve_name_call(self, name: str) -> Tuple[str, Optional[str], bool, str]:
        """Resolve a direct name call like func()."""
        # Check local functions first
        if name in self._local_functions:
            func_info = self._local_functions[name]
            qualname = f"{self._get_caller_module()}.{name}"
            return (name, qualname, True, "local")

        # Check imported names (from imports)
        if name in self._from_imports:
            module = self._from_imports[name]
            qualname = f"{module}.{name}"
            return (name, qualname, True, "imported")

        # Check imported modules (import X; X.func())
        # This would be handled by attribute resolution

        # Could be a builtin or external - cannot resolve confidently
        return (name, None, False, "unresolved")

    def _resolve_attribute_call(self, node: ast.Attribute) -> Tuple[str, Optional[str], bool, str]:
        """Resolve an attribute call like obj.method() or module.func()."""
        # Build the full attribute chain
        attr_chain = self._get_attribute_chain(node)

        if not attr_chain:
            return ("<unknown>", None, False, "unresolved")

        base = attr_chain[0]
        method_name = attr_chain[-1]

        # Case 1: self.method() - method call on current class
        if base == "self" and self.current_class:
            # Look up method in current class
            class_info = self._local_classes.get(self.current_class)
            if class_info:
                for method in class_info.methods:
                    if method.name == method_name:
                        qualname = f"{self._get_caller_module()}.{self.current_class}.{method_name}"
                        return (method_name, qualname, True, "method")

        # Case 2: cls.method() or ClassName.method() - class method call
        if base in self._local_classes:
            class_info = self._local_classes[base]
            for method in class_info.methods:
                if method.name == method_name:
                    qualname = f"{self._get_caller_module()}.{base}.{method_name}"
                    return (method_name, qualname, True, "method")

        # Case 3: module.func() - imported module function call
        if base in self._imported_modules:
            qualname = f"{base}.{method_name}"
            return (method_name, qualname, True, "imported")

        # Case 4: alias.func() - imported alias (e.g., np.array())
        if base in self._imported_names:
            module = self._imported_names[base]
            qualname = f"{module}.{method_name}"
            return (method_name, qualname, True, "imported")

        # Case 5: func from from_import (e.g., from os import path; path.join())
        if base in self._from_imports:
            module = self._from_imports[base]
            qualname = f"{module}.{method_name}"
            return (method_name, qualname, True, "imported")

        # Case 6: instance.method() - cannot resolve instance type statically
        # This would require type inference

        return (method_name, None, False, "unresolved")

    def _get_attribute_chain(self, node: ast.Attribute) -> List[str]:
        """Extract attribute chain from an AST Attribute node.

        e.g., for `a.b.c.d()` returns ['a', 'b', 'c', 'd']
        """
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        elif isinstance(current, ast.Call):
            # Function call returning object - cannot resolve
            return []
        elif isinstance(current, ast.Subscript):
            # Indexed access - cannot resolve
            return []
        return list(reversed(parts))

    def _get_caller_module(self) -> str:
        """Get module name for the current file."""
        # We don't have filepath here directly, but we can derive from context
        # The filepath will be set in parse() function
        return "<module>"


def parse(source: str, filepath: Path) -> ParsedFile:
    """Parse Python source code and extract code elements.

    Args:
        source: Python source code as string
        filepath: Path to the file (for error messages and tracking)

    Returns:
        ParsedFile object containing extracted elements or error information

    Example:
        >>> from pathlib import Path
        >>> source = "def hello(): return 'world'"
        >>> parsed = parse(source, Path("example.py"))
        >>> print(parsed.functions[0].name)
        hello
    """
    try:
        tree = ast.parse(source, filename=str(filepath))
        extractor = CodeElementExtractor()
        extractor.visit(tree)

        # Update qualnames in calls with actual module name
        module_name = filepath.with_suffix('').as_posix().replace('/', '.')
        updated_calls = []
        for call in extractor.calls:
            if call.callee_qualname and call.callee_qualname.startswith("<module>."):
                new_qualname = call.callee_qualname.replace("<module>.", f"{module_name}.")
                updated_calls.append(CallInfo(
                    caller_name=call.caller_name,
                    caller_type=call.caller_type,
                    caller_class=call.caller_class,
                    caller_start_line=call.caller_start_line,
                    callee_name=call.callee_name,
                    callee_qualname=new_qualname,
                    call_line=call.call_line,
                    is_resolved=call.is_resolved,
                    resolution_type=call.resolution_type,
                ))
            else:
                updated_calls.append(call)

        return ParsedFile(
            filepath=filepath,
            functions=extractor.functions,
            classes=extractor.classes,
            imports=extractor.imports,
            calls=updated_calls,
            module_docstring=extractor.module_docstring,
            has_syntax_error=False,
            syntax_error_message=None,
        )
    except SyntaxError as e:
        # Return ParsedFile with error flag
        return ParsedFile(
            filepath=filepath,
            functions=[],
            classes=[],
            imports=[],
            module_docstring=None,
            has_syntax_error=True,
            syntax_error_message=f"{e.__class__.__name__}: {e.msg} (line {e.lineno})",
        )
    except Exception as e:
        # Catch any other parsing errors
        return ParsedFile(
            filepath=filepath,
            functions=[],
            classes=[],
            imports=[],
            module_docstring=None,
            has_syntax_error=True,
            syntax_error_message=f"{e.__class__.__name__}: {str(e)}",
        )
