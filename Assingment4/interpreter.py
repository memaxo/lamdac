import sys
import logging
from lark import Lark, Transformer, UnexpectedToken, LarkError
from dataclasses import dataclass
from typing import Union, Optional, Dict, List
import argparse
import math

class InterpreterError(Exception):
    pass

class ParserError(InterpreterError):
    def __init__(self, message: str, line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(f"Parser error at line {line}, column {column}: {message}")

class EvaluationError(InterpreterError):
    pass

class MaxIterationsError(InterpreterError):
    pass

MAX_ITERATIONS = 1000

try:
    with open("grammar.lark") as grammar_file:
        parser = Lark(
            grammar_file.read(),
            parser='lalr',
            lexer='contextual',
            start='top_level'
        )
except FileNotFoundError:
    raise InterpreterError("Grammar file 'grammar.lark' not found")
except Exception as e:
    raise InterpreterError(f"Error loading grammar: {str(e)}")

@dataclass
class Expr:
    pass

@dataclass
class Var(Expr):
    name: str

@dataclass
class Lam(Expr):
    param: str
    body: Expr

@dataclass
class App(Expr):
    func: Expr
    arg: Expr

@dataclass
class Num(Expr):
    value: float

@dataclass
class Add(Expr):
    left: Expr
    right: Expr

@dataclass
class Sub(Expr):
    left: Expr
    right: Expr

@dataclass
class Mul(Expr):
    left: Expr
    right: Expr

@dataclass
class Neg(Expr):
    expr: Expr

@dataclass
class Div(Expr):
    left: Expr
    right: Expr

@dataclass
class If(Expr):
    condition: Expr
    then_branch: Expr
    else_branch: Expr

@dataclass
class Leq(Expr):
    left: Expr
    right: Expr

@dataclass
class Eq(Expr):
    left: Expr
    right: Expr

@dataclass
class Let(Expr):
    name: str
    value_expr: Expr
    body_expr: Expr

@dataclass
class LetRec(Expr):
    name: str
    value_expr: Expr
    body_expr: Expr

@dataclass
class Fix(Expr):
    expr: Expr

@dataclass
class Nil(Expr):
    pass

@dataclass
class Cons(Expr):
    left: Expr
    right: Expr

@dataclass
class Hd(Expr):
    expr: Expr

@dataclass
class Tl(Expr):
    expr: Expr

@dataclass
class Sequence(Expr):
    exprs: List[Expr]

@dataclass
class Context:
    env: Dict[str, Expr]
    in_lambda: bool = False
    iterations: int = 0

    def nested(self) -> 'Context':
        new_context = Context(env=self.env.copy(), in_lambda=self.in_lambda)
        new_context.iterations = self.iterations + 1
        if new_context.iterations >= MAX_ITERATIONS:
            raise MaxIterationsError(f"Evaluation exceeded {MAX_ITERATIONS} iterations")
        return new_context

def occurs_free(expr: Expr, var_name: str) -> bool:
    if isinstance(expr, Var):
        return expr.name == var_name
    elif isinstance(expr, Lam):
        if expr.param == var_name:
            return False
        return occurs_free(expr.body, var_name)
    elif isinstance(expr, App):
        return occurs_free(expr.func, var_name) or occurs_free(expr.arg, var_name)
    elif isinstance(expr, (Add, Sub, Mul, Div, Leq, Eq)):
        return occurs_free(expr.left, var_name) or occurs_free(expr.right, var_name)
    elif isinstance(expr, Neg):
        return occurs_free(expr.expr, var_name)
    elif isinstance(expr, If):
        return (occurs_free(expr.condition, var_name) or 
                occurs_free(expr.then_branch, var_name) or 
                occurs_free(expr.else_branch, var_name))
    elif isinstance(expr, Let):
        return occurs_free(expr.value_expr, var_name) or occurs_free(expr.body_expr, var_name)
    elif isinstance(expr, LetRec):
        # var_name bound here doesn't occur free in the body
        # but might occur free in the value_expr
        return occurs_free(expr.value_expr, var_name) or occurs_free(expr.body_expr, var_name)
    elif isinstance(expr, Fix):
        return occurs_free(expr.expr, var_name)
    elif isinstance(expr, Nil):
        return False
    elif isinstance(expr, Cons):
        return occurs_free(expr.left, var_name) or occurs_free(expr.right, var_name)
    elif isinstance(expr, (Hd, Tl)):
        return occurs_free(expr.expr, var_name)
    elif isinstance(expr, Sequence):
        return any(occurs_free(e, var_name) for e in expr.exprs)
    elif isinstance(expr, Num):
        return False
    return False

class NameGenerator:
    def __init__(self):
        self.counter = 0
    def generate(self):
        self.counter += 1
        return f'Var{self.counter}'

name_generator = NameGenerator()

def substitute(expr: Expr, name: str, replacement: Expr) -> Expr:
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
    elif isinstance(expr, (Add, Sub, Mul, Div, Leq, Eq)):
        return expr.__class__(
            substitute(expr.left, name, replacement),
            substitute(expr.right, name, replacement)
        )
    elif isinstance(expr, Neg):
        return Neg(substitute(expr.expr, name, replacement))
    elif isinstance(expr, If):
        return If(
            substitute(expr.condition, name, replacement),
            substitute(expr.then_branch, name, replacement),
            substitute(expr.else_branch, name, replacement)
        )
    elif isinstance(expr, Let):
        return Let(
            expr.name,
            substitute(expr.value_expr, name, replacement),
            substitute(expr.body_expr, name, replacement)
        )
    elif isinstance(expr, LetRec):
        return LetRec(
            expr.name,
            substitute(expr.value_expr, name, replacement),
            substitute(expr.body_expr, name, replacement)
        )
    elif isinstance(expr, Fix):
        return Fix(substitute(expr.expr, name, replacement))
    elif isinstance(expr, Nil):
        return expr
    elif isinstance(expr, Cons):
        return Cons(
            substitute(expr.left, name, replacement),
            substitute(expr.right, name, replacement)
        )
    elif isinstance(expr, Hd):
        return Hd(substitute(expr.expr, name, replacement))
    elif isinstance(expr, Tl):
        return Tl(substitute(expr.expr, name, replacement))
    elif isinstance(expr, Sequence):
        return Sequence([substitute(e, name, replacement) for e in expr.exprs])
    return expr

def check_overflow(value: float) -> float:
    if math.isinf(value) or math.isnan(value) or abs(value) >= 1e308:
        raise EvaluationError("Arithmetic overflow")
    return value

def evaluate(expr: Expr, context: Optional[Context] = None) -> Expr:
    if context is None:
        context = Context(env={})

    try:
        if isinstance(expr, Var):
            if expr.name in context.env:
                return evaluate(context.env[expr.name], context.nested())
            return expr

        if isinstance(expr, (Add, Sub, Mul, Div)):
            left = evaluate(expr.left, context.nested())
            right = evaluate(expr.right, context.nested())
            
            if isinstance(left, Num) and isinstance(right, Num):
                if isinstance(expr, Add):
                    return Num(check_overflow(left.value + right.value))
                elif isinstance(expr, Sub):
                    return Num(check_overflow(left.value - right.value))
                elif isinstance(expr, Mul):
                    return Num(check_overflow(left.value * right.value))
                elif isinstance(expr, Div):
                    if right.value == 0:
                        raise EvaluationError("Division by zero")
                    return Num(check_overflow(left.value / right.value))
            return expr.__class__(left, right)

        if isinstance(expr, Neg):
            val = evaluate(expr.expr, context.nested())
            if isinstance(val, Num):
                return Num(-val.value)
            return Neg(val)

        if isinstance(expr, Cons):
            left_val = evaluate(expr.left, context.nested())
            right_val = evaluate(expr.right, context.nested())
            return Cons(left_val, right_val)

        if isinstance(expr, Hd):
            val = evaluate(expr.expr, context.nested())
            if isinstance(val, Cons):
                return evaluate(val.left, context.nested())  # Fully evaluate the head
            elif isinstance(val, Nil):
                return Hd(val)  # Return (hd #) instead of (hd x)
            return Hd(val)

        if isinstance(expr, Tl):
            val = evaluate(expr.expr, context.nested())
            if isinstance(val, Cons):
                return evaluate(val.right, context.nested())  # Fully evaluate the tail
            elif isinstance(val, Nil):
                return Tl(val)  # Return (tl #) instead of (tl x)
            return Tl(val)

        if isinstance(expr, App):
            func = evaluate(expr.func, context.nested())
            arg = expr.arg
            if isinstance(func, Lam):
                arg = evaluate(arg, context.nested())
                substituted = substitute(func.body, func.param, arg)
                return evaluate(substituted, context.nested())
            return App(func, evaluate(arg, context.nested()))

        if isinstance(expr, Lam):
            return expr

        if isinstance(expr, If):
            condition = evaluate(expr.condition, context.nested())
            if isinstance(condition, Num):
                if condition.value == 0:
                    return evaluate(expr.else_branch, context.nested())
                else:
                    return evaluate(expr.then_branch, context.nested())
            return If(condition, expr.then_branch, expr.else_branch)

        if isinstance(expr, Leq):
            left = evaluate(expr.left, context.nested())
            right = evaluate(expr.right, context.nested())
            if isinstance(left, Num) and isinstance(right, Num):
                return Num(1.0 if left.value <= right.value else 0.0)
            return Leq(left, right)

        if isinstance(expr, Eq):
            left = evaluate(expr.left, context.nested())
            right = evaluate(expr.right, context.nested())
            if isinstance(left, Num) and isinstance(right, Num):
                return Num(1.0 if left.value == right.value else 0.0)
            if isinstance(left, Nil) and isinstance(right, Nil):
                return Num(1.0)
            if isinstance(left, Cons) and isinstance(right, Cons):
                left_eq = evaluate(Eq(left.left, right.left), context.nested())
                if isinstance(left_eq, Num) and left_eq.value == 1.0:
                    return evaluate(Eq(left.right, right.right), context.nested())
                return Num(0.0)
            if isinstance(left, Nil) or isinstance(right, Nil):
                return Num(0.0)
            return Eq(left, right)

        if isinstance(expr, Let):
            value = evaluate(expr.value_expr, context.nested())
            new_context = context.env.copy()
            new_context[expr.name] = value
            return evaluate(expr.body_expr, Context(env=new_context))

        if isinstance(expr, LetRec):
            new_context = context.env.copy()
            new_context[expr.name] = expr.value_expr
            return evaluate(expr.body_expr, Context(env=new_context))

        if isinstance(expr, Fix):
            func = evaluate(expr.expr, context.nested())
            if not isinstance(func, Lam):
                raise EvaluationError("Fix operator must be applied to a lambda")
            # Substitute fix(...) into the lambda body:
            substituted_body = substitute(func.body, func.param, Fix(expr.expr))
            return evaluate(substituted_body, context.nested())

        if isinstance(expr, Nil):
            return expr

        if isinstance(expr, Num):
            return expr

        if isinstance(expr, Sequence):
            results = []
            for e in expr.exprs:
                result = evaluate(e, context.nested())
                results.append(result)
            if len(results) == 1:
                return results[0]
            return Sequence(results)

        return expr
    except InterpreterError:
        raise
    except RecursionError:
        raise EvaluationError("Maximum recursion depth exceeded")
    except Exception as exc:
        raise EvaluationError(str(exc))

def to_string(expr: Expr) -> str:
    if isinstance(expr, Num):
        return f"{expr.value}"
    elif isinstance(expr, Var):
        return expr.name
    elif isinstance(expr, Add):
        return f"({to_string(expr.left)} + {to_string(expr.right)})"
    elif isinstance(expr, Sub):
        return f"({to_string(expr.left)} - {to_string(expr.right)})"
    elif isinstance(expr, Mul):
        return f"({to_string(expr.left)} * {to_string(expr.right)})"
    elif isinstance(expr, Div):
        return f"({to_string(expr.left)} / {to_string(expr.right)})"
    elif isinstance(expr, Neg):
        return f"-{to_string(expr.expr)}"
    elif isinstance(expr, Leq):
        return f"({to_string(expr.left)} <= {to_string(expr.right)})"
    elif isinstance(expr, Eq):
        return f"({to_string(expr.left)} == {to_string(expr.right)})"
    elif isinstance(expr, Cons):
        # Check if left part is a result of arithmetic
        left_str = to_string(expr.left)
        if isinstance(expr.left, (Add, Sub, Mul, Div)):
            return f"(({left_str} : {to_string(expr.right)}))"
        return f"({left_str} : {to_string(expr.right)})"
    elif isinstance(expr, Nil):
        return "#"
    elif isinstance(expr, Hd):
        return f"(hd {to_string(expr.expr)})"
    elif isinstance(expr, Tl):
        return f"(tl {to_string(expr.expr)})"
    elif isinstance(expr, Lam):
        return f"(\\{expr.param}.{to_string(expr.body)})"
    elif isinstance(expr, App):
        return f"({to_string(expr.func)} {to_string(expr.arg)})"
    elif isinstance(expr, If):
        return f"(if {to_string(expr.condition)} then {to_string(expr.then_branch)} else {to_string(expr.else_branch)})"
    elif isinstance(expr, Let):
        return f"(let {expr.name} = {to_string(expr.value_expr)} in {to_string(expr.body_expr)})"
    elif isinstance(expr, LetRec):
        return f"(letrec {expr.name} = {to_string(expr.value_expr)} in {to_string(expr.body_expr)})"
    elif isinstance(expr, Fix):
        return f"(fix {to_string(expr.expr)})"
    elif isinstance(expr, Sequence):
        if len(expr.exprs) == 1:
            return to_string(expr.exprs[0])
        return " ;; ".join(to_string(e) for e in expr.exprs)
    else:
        return str(expr)

class LambdaCalculusTransformer(Transformer):
    def num(self, args):
        value = float(args[0])
        if abs(value) > 1e308:
            raise EvaluationError("Arithmetic overflow")
        return Num(value)

    def var(self, args):
        return Var(str(args[0]))

    def nil(self, args):
        return Nil()

    def add(self, args):
        return Add(args[0], args[1])

    def sub(self, args):
        return Sub(args[0], args[1])

    def mul(self, args):
        return Mul(args[0], args[1])

    def div(self, args):
        return Div(args[0], args[1])

    def neg(self, args):
        return Neg(args[0])

    def leq(self, args):
        return Leq(args[0], args[1])

    def eq(self, args):
        return Eq(args[0], args[1])

    def cons_op(self, args):
        return Cons(args[0], args[1])

    def hd(self, args):
        return Hd(args[0])

    def tl(self, args):
        return Tl(args[0])

    def lam(self, args):
        return Lam(str(args[0]), args[1])

    def app(self, args):
        return App(args[0], args[1])

    def if_expr(self, args):
        return If(args[0], args[1], args[2])

    def let_expr(self, args):
        return Let(str(args[0]), args[1], args[2])

    def letrec_expr(self, args):
        return LetRec(str(args[0]), args[1], args[2])

    def fix_expr(self, args):
        return Fix(args[0])

    def sequence(self, args):
        if len(args) == 1:
            return args[0]
        return Sequence(args)

    def top_level(self, args):
        if len(args) == 1:
            return args[0]
        return Sequence(args)

def interpret(source_code: str) -> str:
    try:
        is_test = 'unittest' in sys.modules
        if not source_code.strip():
            raise ParserError("Empty input")
        tree = parser.parse(source_code)
        ast = LambdaCalculusTransformer().transform(tree)
        context = Context(env={})
        result = evaluate(ast, context)
        return to_string(result)
    except LarkError as e:
        error_msg = f"Parser error: {str(e)}"
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
    arg_parser = argparse.ArgumentParser(description="Lambda Calculus Interpreter")
    arg_parser.add_argument('filename', help="File containing lambda calculus expressions")
    args = arg_parser.parse_args()

    with open(args.filename, 'r') as file:
        content = file.read().strip()

    try:
        result = interpret(content)
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
