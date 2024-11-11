import sys
from lark import Lark, Transformer, UnexpectedInput, UnexpectedToken, UnexpectedCharacters, Token
from dataclasses import dataclass
from typing import Union, Optional, Dict, Set
from enum import Enum
import argparse
import logging
import re

class InterpreterError(Exception):
    """Base class for interpreter errors"""
    pass

class ParserError(InterpreterError):
    """Error during parsing phase"""
    def __init__(self, message: str, line: Optional[int] = None, column: Optional[int] = None):
        self.line = line
        self.column = column
        super().__init__(f"Parser error at line {line}, column {column}: {message}")

class EvaluationError(InterpreterError):
    """Error during evaluation phase"""
    pass

class SubstitutionError(InterpreterError):
    """Error during variable substitution"""
    pass

class MaxIterationsError(InterpreterError):
    """Maximum iteration limit exceeded"""
    pass

# Maximum number of iterations to prevent infinite recursion
MAX_ITERATIONS = 1000

# Initialize the parser with the grammar from 'grammar.lark'
try:
    with open("grammar.lark") as grammar_file:
        parser = Lark(
            grammar_file.read(),
            parser='lalr',
            lexer='standard',
            start='start'
        )
except FileNotFoundError:
    raise InterpreterError("Grammar file 'grammar.lark' not found")
except Exception as e:
    raise InterpreterError(f"Error loading grammar: {str(e)}")

# Data classes for different types of expressions
@dataclass
class Expr:
    pass

@dataclass
class Var(Expr):
    """Variable expression."""
    name: str

@dataclass
class Lam(Expr):
    """Lambda abstraction."""
    param: str
    body: Expr

@dataclass
class App(Expr):
    """Function application."""
    func: Expr
    arg: Expr

@dataclass
class Num(Expr):
    """Numeric literal."""
    value: float

@dataclass
class Add(Expr):
    """Addition operation."""
    left: Expr
    right: Expr

@dataclass
class Sub(Expr):
    """Subtraction operation."""
    left: Expr
    right: Expr

@dataclass
class Mul(Expr):
    """Multiplication operation."""
    left: Expr
    right: Expr

@dataclass
class Neg(Expr):
    """Negation operation."""
    expr: Expr

@dataclass
class Div(Expr):
    """Division operation."""
    left: Expr
    right: Expr

@dataclass
class Context:
    """Evaluation context for expressions."""
    env: Dict[str, Expr]
    in_lambda: bool = False
    iterations: int = 0

    def nested(self) -> 'Context':
        """Creates a new context for nested evaluation."""
        return Context(
            env=self.env.copy(),
            in_lambda=self.in_lambda,
            iterations=self.iterations + 1
        )

    def inside_lambda(self) -> 'Context':
        """Creates a new context inside a lambda."""
        return Context(
            env=self.env.copy(),
            in_lambda=True,
            iterations=self.iterations
        )

class LambdaCalculusTransformer(Transformer):
    """Transforms parsed tokens into expression objects."""

    def lam(self, args):
        """Handle lambda abstraction: LAMBDA NAME DOT expression"""
        return Lam(str(args[1]), args[3])

    def app(self, args):
        """Handle function application"""
        return App(args[0], args[1])

    def var(self, args):
        """Handle variables"""
        return Var(str(args[0]))

    def num(self, args):
        """Handle numeric literals"""
        value = float(args[0])
        return Num(value)

    def add(self, args):
        """Handle addition: expression PLUS application"""
        return Add(args[0], args[2])

    def sub(self, args):
        """Handle subtraction: expression MINUS application"""
        return Sub(args[0], args[2])

    def mul(self, args):
        """Handle multiplication: application TIMES factor"""
        return Mul(args[0], args[2])

    def div(self, args):
        """Handle division: application DIVIDE factor"""
        return Div(args[0], args[2])

    def neg(self, args):
        """Handle negation: MINUS factor"""
        return Neg(args[1])

    def group(self, args):
        """Handle parenthesized expressions"""
        return args[1]

    def __default__(self, data, children, meta):
        """Handle any unhandled rules"""
        if len(children) == 1:
            return children[0]
        return children


class NameGenerator:
    """Generates unique variable names to avoid variable capture."""

    def __init__(self):
        self.counter = 0

    def generate(self):
        self.counter += 1
        return f'Var{self.counter}'

name_generator = NameGenerator()

def check_overflow(value: float) -> float:
    """Check if a numeric value would cause overflow."""
    if abs(value) > 1e308:
        raise EvaluationError("Arithmetic overflow")
    return value

def evaluate(expr: Expr, context: Optional[Context] = None) -> Expr:
    if context is None:
        context = Context(env={})

    if context.iterations >= MAX_ITERATIONS:
        raise MaxIterationsError(f"Evaluation exceeded {MAX_ITERATIONS} iterations")

    try:
        if isinstance(expr, Lam):
            return expr

        if isinstance(expr, App):
            func = evaluate(expr.func, context.nested())
            if isinstance(func, Lam):
                arg = evaluate(expr.arg, context.nested())
                substituted = substitute(func.body, func.param, arg)
                return evaluate(substituted, context.nested())
            else:
                arg = evaluate(expr.arg, context.nested())
                return App(func, arg)

        if isinstance(expr, (Add, Sub, Mul, Div)):
            left = evaluate(expr.left, context)
            right = evaluate(expr.right, context)
            
            # Handle arithmetic with variables in lambda bodies
            if context.in_lambda and (isinstance(left, Var) or isinstance(right, Var)):
                return expr.__class__(left, right)
            
            # Evaluate nested expressions first
            if isinstance(left, (Add, Sub, Mul, Div, App)) or isinstance(right, (Add, Sub, Mul, Div, App)):
                return expr.__class__(left, right)
            
            if isinstance(left, Num) and isinstance(right, Num):
                if isinstance(expr, Add):
                    result = check_overflow(left.value + right.value)
                elif isinstance(expr, Sub):
                    result = check_overflow(left.value - right.value)
                elif isinstance(expr, Mul):
                    # Handle multiplication with proper sign handling
                    result = check_overflow(left.value * right.value)
                elif isinstance(expr, Div):
                    if abs(right.value) < 1e-10:
                        raise EvaluationError("Division by zero")
                    result = check_overflow(left.value / right.value)
                return Num(result)
            return expr.__class__(left, right)

        if isinstance(expr, Neg):
            value = evaluate(expr.expr, context)
            if isinstance(value, Num):
                return Num(-value.value)
            # If the value is not a number, maintain the negation
            return Neg(value)

        if isinstance(expr, Num):
            return expr

        if isinstance(expr, Var):
            if expr.name in context.env:
                return evaluate(context.env[expr.name], context)
            return expr

        return expr

    except InterpreterError:
        raise
    except Exception as e:
        raise EvaluationError(f"Evaluation error: {str(e)}")

def evaluate_arithmetic(expr: Expr, context: Context) -> Expr:
    """Evaluate arithmetic expressions."""
    if isinstance(expr, Add):
        left = evaluate(expr.left, context)
        right = evaluate(expr.right, context)
        
        # Allow arithmetic with variables inside lambda bodies
        if context.in_lambda and (isinstance(left, Var) or isinstance(right, Var)):
            return Add(left, right)
            
        if isinstance(left, Num) and isinstance(right, Num):
            result = check_overflow(left.value + right.value)
            return Num(result)
        return Add(left, right)
        
    elif isinstance(expr, Sub):
        left = evaluate(expr.left, context)
        right = evaluate(expr.right, context)
        
        if context.in_lambda and (isinstance(left, Var) or isinstance(right, Var)):
            return Sub(left, right)
            
        if isinstance(left, Num) and isinstance(right, Num):
            result = check_overflow(left.value - right.value)
            return Num(result)
        return Sub(left, right)
        
    elif isinstance(expr, Mul):
        left = evaluate(expr.left, context)
        right = evaluate(expr.right, context)
        
        if context.in_lambda and (isinstance(left, Var) or isinstance(right, Var)):
            return Mul(left, right)
            
        if isinstance(left, Num) and isinstance(right, Num):
            result = check_overflow(left.value * right.value)
            return Num(result)
        return Mul(left, right)
        
    elif isinstance(expr, Div):
        left = evaluate(expr.left, context)
        right = evaluate(expr.right, context)
        
        if context.in_lambda and (isinstance(left, Var) or isinstance(right, Var)):
            return Div(left, right)
            
        if isinstance(left, Num) and isinstance(right, Num):
            if abs(right.value) < 1e-10:
                raise EvaluationError("Division by zero")
            result = check_overflow(left.value / right.value)
            return Num(result)
        return Div(left, right)
        
    elif isinstance(expr, Neg):
        value = evaluate(expr.expr, context)
        if isinstance(value, Num):
            result = check_overflow(-value.value)
            return Num(result)
        return Neg(value)
        
    else:
        raise EvaluationError(f"Unknown arithmetic expression: {expr}")

def substitute(expr: Expr, name: str, replacement: Expr) -> Expr:
    """
    Performs capture-avoiding substitution while preserving structure.
    """
    if isinstance(expr, Var):
        return replacement if expr.name == name else expr
    elif isinstance(expr, Lam):
        if expr.param == name:
            return expr
        elif occurs_free(replacement, expr.param):
            fresh_name = name_generator.generate()
            new_body = substitute(expr.body, expr.param, Var(fresh_name))
            return Lam(fresh_name, substitute(new_body, name, replacement))
        else:
            return Lam(expr.param, substitute(expr.body, name, replacement))
    elif isinstance(expr, App):
        return App(substitute(expr.func, name, replacement),
                   substitute(expr.arg, name, replacement))
    elif isinstance(expr, (Add, Sub, Mul, Div)):
        return expr.__class__(
            substitute(expr.left, name, replacement),
            substitute(expr.right, name, replacement)
        )
    elif isinstance(expr, Neg):
        return Neg(substitute(expr.expr, name, replacement))
    else:
        return expr

def occurs_free(expr: Expr, var_name: str) -> bool:
    """
    Checks if the variable name occurs free in the expression.
    """
    if isinstance(expr, Var):
        return expr.name == var_name
    elif isinstance(expr, Lam):
        if expr.param == var_name:
            return False
        else:
            return occurs_free(expr.body, var_name)
    elif isinstance(expr, App):
        return occurs_free(expr.func, var_name) or occurs_free(expr.arg, var_name)
    elif isinstance(expr, (Add, Sub, Mul, Div)):
        return occurs_free(expr.left, var_name) or occurs_free(expr.right, var_name)
    elif isinstance(expr, Neg):
        return occurs_free(expr.expr, var_name)
    else:
        return False

def linearize(expr: Expr) -> str:
    """Converts the expression back to a string representation."""
    if isinstance(expr, Var):
        return expr.name
    elif isinstance(expr, Lam):
        return f"(\\{expr.param}.{linearize(expr.body)})"
    elif isinstance(expr, App):
        return f"({linearize(expr.func)} {linearize(expr.arg)})"
    elif isinstance(expr, Num):
        return f"{expr.value:.1f}"
    elif isinstance(expr, Add):
        return f"({linearize(expr.left)} + {linearize(expr.right)})"
    elif isinstance(expr, Sub):
        return f"({linearize(expr.left)} - {linearize(expr.right)})"
    elif isinstance(expr, Mul):
        return f"({linearize(expr.left)} * {linearize(expr.right)})"
    elif isinstance(expr, Div):
        return f"({linearize(expr.left)} / {linearize(expr.right)})"
    elif isinstance(expr, Neg):
        inner = linearize(expr.expr)
        if isinstance(expr.expr, Num):
            return f"-{inner}"  # No extra parentheses for numbers
        return f"(-{inner})"
    else:
        raise Exception(f"Unknown expression type: {type(expr)}")

def get_free_variables(expr: Expr) -> Set[str]:
    """Returns the set of free variables in an expression."""
    def collect(e: Expr, bound: Set[str]) -> Set[str]:
        if isinstance(e, Var):
            return {e.name} if e.name not in bound else set()
        elif isinstance(e, Lam):
            return collect(e.body, bound | {e.param})
        elif isinstance(e, App):
            return collect(e.func, bound) | collect(e.arg, bound)
        elif isinstance(e, (Add, Sub, Mul, Div)):
            return collect(e.left, bound) | collect(e.right, bound)
        elif isinstance(e, Neg):
            return collect(e.expr, bound)
        return set()

    return collect(expr, set())

def interpret(source_code: str) -> str:
    try:
        if not source_code or not source_code.strip():
            raise ParserError("Empty input", None, None)

        # Parse and transform
        try:
            cst = parser.parse(source_code)
        except UnexpectedToken as e:
            raise ParserError(f"Syntax error: {str(e)}", e.line, e.column)

        transformer = LambdaCalculusTransformer()
        ast = transformer.transform(cst)

        # Evaluate eagerly
        result = evaluate(ast)

        return linearize(result)

    except InterpreterError as e:
        logging.error(f"Error interpreting: {source_code}\n{str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise InterpreterError(f"Internal error: {str(e)}")

def main():
    """
    Main function to interpret expressions from a file provided as a command-line argument.
    """
    arg_parser = argparse.ArgumentParser(description="Lambda Calculus Interpreter")
    arg_parser.add_argument('filename', help="File containing lambda calculus expressions")
    args = arg_parser.parse_args()

    with open(args.filename, 'r') as file:
        expressions = file.read().split('\n')

    for expression in expressions:
        expression = expression.strip()
        if expression and not expression.startswith('//'):
            try:
                result = interpret(expression)
                print(f"Expression: {expression}")
                print(f"Result: \033[95m{result}\033[0m")
                print()
            except Exception as e:
                print(f"Failed to interpret expression: {expression}")
                print(f"Error: {str(e)}\n")

if __name__ == "__main__":
    main()
