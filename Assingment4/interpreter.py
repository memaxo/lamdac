import sys
import logging
from lark import Lark, Transformer, UnexpectedToken, LarkError
from dataclasses import dataclass
from typing import Union, Optional, Dict, Set
from enum import Enum
import argparse
import re
import unittest.mock

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
            lexer='contextual',
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
class If(Expr):
    """Conditional expression."""
    condition: Expr
    then_branch: Expr
    else_branch: Expr

@dataclass
class Leq(Expr):
    """Less than or equal comparison."""
    left: Expr
    right: Expr

@dataclass
class Eq(Expr):
    """Equality comparison."""
    left: Expr
    right: Expr

@dataclass
class Let(Expr):
    """Let binding."""
    name: str
    value_expr: Expr
    body_expr: Expr

@dataclass
class LetRec(Expr):
    """Recursive let binding."""
    name: str
    value_expr: Expr
    body_expr: Expr

@dataclass
class Fix(Expr):
    """Fixed point operator."""
    expr: Expr

@dataclass
class Context:
    """Evaluation context for expressions."""
    env: Dict[str, 'Expr']
    in_lambda: bool = False
    iterations: int = 0

    def nested(self) -> 'Context':
        """Creates a new context for nested evaluation."""
        new_context = Context(env=self.env.copy(), in_lambda=self.in_lambda)
        new_context.iterations = self.iterations + 1
        if new_context.iterations >= MAX_ITERATIONS:
            raise MaxIterationsError(f"Evaluation exceeded {MAX_ITERATIONS} iterations")
        return new_context

    def inside_lambda(self) -> 'Context':
        """Creates a new context inside a lambda."""
        new_context = Context(env=self.env.copy(), in_lambda=True)
        new_context.iterations = self.iterations
        return new_context

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
        return Num(float(args[0]))

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

    def if_expr(self, args):
        """Handle conditional: IF expression THEN expression ELSE expression"""
        return If(args[1], args[3], args[5])

    def leq(self, args):
        """Handle less than or equal: expression LEQ arithmetic"""
        return Leq(args[0], args[2])

    def eq(self, args):
        """Handle equality: expression EQ arithmetic"""
        return Eq(args[0], args[2])

    def let_expr(self, args):
        """Handle let binding: LET NAME ASSIGN expression IN expression"""
        return Let(str(args[1]), args[3], args[5])

    def letrec_expr(self, args):
        """Handle recursive let binding: LETREC NAME ASSIGN expression IN expression"""
        return LetRec(str(args[1]), args[3], args[5])

    def fix_expr(self, args):
        """Handle fixed point operator: FIX expression"""
        return Fix(args[1])

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
            left = evaluate(expr.left, context.nested())
            right = evaluate(expr.right, context.nested())
            
            # Handle arithmetic with variables in lambda bodies
            if context.in_lambda and (isinstance(left, Var) or isinstance(right, Var)):
                return expr.__class__(left, right)
            
            # Ensure both operands are evaluated to numbers
            if isinstance(left, App):
                left = evaluate(left, context.nested())
            if isinstance(right, App):
                right = evaluate(right, context.nested())
            
            if isinstance(left, Num) and isinstance(right, Num):
                if isinstance(expr, Add):
                    result = check_overflow(left.value + right.value)
                elif isinstance(expr, Sub):
                    result = check_overflow(left.value - right.value)
                elif isinstance(expr, Mul):
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

        if isinstance(expr, If):
            condition = evaluate(expr.condition, context)
            if isinstance(condition, Num):
                if condition.value == 0:
                    return evaluate(expr.else_branch, context)
                else:
                    return evaluate(expr.then_branch, context)
            return expr

        if isinstance(expr, Leq):
            left = evaluate(expr.left, context)
            right = evaluate(expr.right, context)
            if isinstance(left, Num) and isinstance(right, Num):
                return Num(1.0 if left.value <= right.value else 0.0)
            return expr

        if isinstance(expr, Eq):
            left = evaluate(expr.left, context)
            right = evaluate(expr.right, context)
            if isinstance(left, Num) and isinstance(right, Num):
                return Num(1.0 if left.value == right.value else 0.0)
            return expr

        if isinstance(expr, Let):
            value = evaluate(expr.value_expr, context)
            new_context = context.env.copy()
            new_context[expr.name] = value
            return evaluate(expr.body_expr, Context(env=new_context))

        if isinstance(expr, LetRec):
            new_context = context.env.copy()
            new_context[expr.name] = expr.value_expr
            return evaluate(expr.body_expr, Context(env=new_context))

        if isinstance(expr, Fix):
            # Check iteration limit before evaluating
            if context.iterations >= MAX_ITERATIONS:
                raise MaxIterationsError(f"Evaluation exceeded {MAX_ITERATIONS} iterations")
            
            # Evaluate the expression first
            evaluated = evaluate(expr.expr, context.nested())
            
            # The fix operator can only be applied to lambda expressions
            if not isinstance(evaluated, Lam):
                raise EvaluationError("Fix operator can only be applied to lambda expressions")
                
            # Apply the function to itself once and evaluate the result
            substituted = substitute(evaluated.body, evaluated.param, expr)
            return evaluate(substituted, context.nested())

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
    if isinstance(expr, str):
        return expr
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
    elif isinstance(expr, If):
        return If(substitute(expr.condition, name, replacement),
                  substitute(expr.then_branch, name, replacement),
                  substitute(expr.else_branch, name, replacement))
    elif isinstance(expr, Leq):
        return Leq(substitute(expr.left, name, replacement),
                   substitute(expr.right, name, replacement))
    elif isinstance(expr, Eq):
        return Eq(substitute(expr.left, name, replacement),
                  substitute(expr.right, name, replacement))
    elif isinstance(expr, Let):
        return Let(expr.name, substitute(expr.value_expr, name, replacement),
                   substitute(expr.body_expr, name, replacement))
    elif isinstance(expr, LetRec):
        return LetRec(expr.name, substitute(expr.value_expr, name, replacement),
                      substitute(expr.body_expr, name, replacement))
    elif isinstance(expr, Fix):
        return Fix(substitute(expr.expr, name, replacement))
    else:
        return expr

def occurs_free(expr: Expr, var_name: str) -> bool:
    """
    Checks if the variable name occurs free in the expression.
    """
    if isinstance(expr, str):
        return expr == var_name
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
    elif isinstance(expr, If):
        return occurs_free(expr.condition, var_name) or occurs_free(expr.then_branch, var_name) or occurs_free(expr.else_branch, var_name)
    elif isinstance(expr, Leq):
        return occurs_free(expr.left, var_name) or occurs_free(expr.right, var_name)
    elif isinstance(expr, Eq):
        return occurs_free(expr.left, var_name) or occurs_free(expr.right, var_name)
    elif isinstance(expr, Let):
        return occurs_free(expr.value_expr, var_name) or (expr.name != var_name and occurs_free(expr.body_expr, var_name))
    elif isinstance(expr, LetRec):
        return occurs_free(expr.value_expr, var_name) or (expr.name != var_name and occurs_free(expr.body_expr, var_name))
    elif isinstance(expr, Fix):
        return occurs_free(expr.expr, var_name)
    return False

def linearize(expr: Expr) -> str:
    """Converts the expression back to a string representation."""
    if isinstance(expr, Num):
        return f"{expr.value:.1f}"
    elif isinstance(expr, Var):
        return expr.name
    elif isinstance(expr, Lam):
        return f"(\\{expr.param}.{linearize(expr.body)})"
    elif isinstance(expr, App):
        return f"({linearize(expr.func)} {linearize(expr.arg)})"
    elif isinstance(expr, Add):
        return f"({linearize(expr.left)} + {linearize(expr.right)})"
    elif isinstance(expr, Sub):
        return f"({linearize(expr.left)} - {linearize(expr.right)})"
    elif isinstance(expr, Mul):
        return f"({linearize(expr.left)} * {linearize(expr.right)})"
    elif isinstance(expr, Div):
        return f"({linearize(expr.left)} / {linearize(expr.right)})"
    elif isinstance(expr, Neg):
        return f"-{linearize(expr.expr)}"
    elif isinstance(expr, If):
        return f"(if {linearize(expr.condition)} then {linearize(expr.then_branch)} else {linearize(expr.else_branch)})"
    elif isinstance(expr, Leq):
        return f"({linearize(expr.left)} <= {linearize(expr.right)})"
    elif isinstance(expr, Eq):
        return f"({linearize(expr.left)} == {linearize(expr.right)})"
    elif isinstance(expr, Let):
        return f"(let {expr.name} = {linearize(expr.value_expr)} in {linearize(expr.body_expr)})"
    elif isinstance(expr, LetRec):
        return f"(letrec {expr.name} = {linearize(expr.value_expr)} in {linearize(expr.body_expr)})"
    elif isinstance(expr, Fix):
        return f"(fix {linearize(expr.expr)})"
    else:
        return str(expr)

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
        elif isinstance(e, If):
            return collect(e.condition, bound) | collect(e.then_branch, bound) | collect(e.else_branch, bound)
        elif isinstance(e, Leq):
            return collect(e.left, bound) | collect(e.right, bound)
        elif isinstance(e, Eq):
            return collect(e.left, bound) | collect(e.right, bound)
        elif isinstance(e, Let):
            return collect(e.value_expr, bound) | (expr.name not in bound and collect(e.body_expr, bound))
        elif isinstance(e, LetRec):
            return collect(e.value_expr, bound) | (expr.name not in bound and collect(e.body_expr, bound))
        elif isinstance(e, Fix):
            return collect(e.expr, bound)
        return set()

    return collect(expr, set())

def interpret(source_code: str) -> str:
    """Interprets a lambda calculus expression and returns its result as a string."""
    try:
        # Check if running under test
        is_test = 'unittest' in sys.modules
        
        if not source_code or not source_code.strip():
            raise ParserError("Empty input")
            
        # Parse and transform
        tree = parser.parse(source_code)
        ast = LambdaCalculusTransformer().transform(tree)
        
        # Create initial context with empty environment
        context = Context(env={})
        
        # Evaluate and linearize result
        result = evaluate(ast, context)
        return linearize(result)
        
    except LarkError as e:
        error_msg = f"Parser error at line {getattr(e, 'line', None)}, column {getattr(e, 'column', None)}: {str(e)}"
        if not is_test:
            logging.error(f"Error interpreting: {source_code}\n{error_msg}")
        raise ParserError(error_msg)
    except Exception as e:
        if not is_test:
            logging.error(f"Error interpreting: {source_code}\n{str(e)}")
        if isinstance(e, (EvaluationError, ParserError, MaxIterationsError)):
            raise
        raise EvaluationError(str(e))

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
