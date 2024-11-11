# Lambda Calculus Interpreter in Python

A sophisticated implementation of lambda calculus with arithmetic operations, demonstrating key concepts in programming language theory and interpreter design.

## Installation

Requirements: `python` (`python3`) and `pip` (`pip3`).

Install with `source setup.sh`. Then `python interpreter_test.py` should pass all tests. You can run your own program in `test.lc` with `python interpreter.py test.lc`. 

## Project Structure

### 1. Grammar Definition (grammar.lark)

The grammar file defines our language's syntax using Lark parser combinators. Notable features:

- Elegant handling of operator precedence through rule layering:
  ```lark
  ?exp: lambda_abs | add_sub
  ?add_sub: add_sub PLUS mul | add_sub MINUS mul | mul
  ?mul: mul TIMES application | application
  ?application: application atom | atom
  ```
- Support for lambda abstractions, arithmetic, and function application
- Careful handling of whitespace and comments
- Proper associativity rules for function application and operators

### 2. Core Interpreter (interpreter.py)

The interpreter implements the evaluation engine with several sophisticated features:

- **AST Implementation**: Uses Python dataclasses for a clean, typed AST representation
- **Evaluation Strategy**: 
  - Implements call-by-name evaluation
  - Lazy evaluation under lambda abstractions
  - Proper handling of arithmetic operations
- **Key Components**:
  ```python
  def evaluate(expr: Expr, iterations: int = 0) -> Expr:
      # Handles beta reduction and arithmetic evaluation
  
  def substitute(expr: Expr, name: str, replacement: Expr) -> Expr:
      # Implements capture-avoiding substitution
  ```

### 3. Test Suite (interpreter_test.py)

Comprehensive test suite demonstrating and verifying language features:

- Basic lambda calculus reduction
- Arithmetic evaluation
- Variable capture avoidance
- Operator precedence
- Lazy evaluation behavior

## Language Features

### Lambda Calculus
- Standard lambda abstraction: `\x.e`
- Function application: `f x`
- Variable references
- Proper capture-avoiding substitution

### Arithmetic Operations
- Addition: `+`
- Multiplication: `*`
- Subtraction: `-`
- Negation: `-x`

### Evaluation Rules

1. **Beta Reduction**: `(\x.e) v → e[v/x]`
2. **Arithmetic**: Evaluates arithmetic expressions when operands are numeric
3. **Lazy Evaluation**: Expressions under lambdas remain unevaluated
4. **Application**: Left-associative function application

### Syntax Notes

- Function application: `a b c` = `(a b) c`
- Lambda abstraction: `\a.\b.c d` = `\a.(\b.c d)`
- Variable names must start with lowercase letters
- `Var1`, `Var2`, etc. are reserved for fresh variable generation
- Comments start with `//`

## Implementation Details

### Parser Workflow
1. Lark parser converts source to concrete syntax tree
2. LambdaCalculusTransformer converts to typed AST
3. Evaluator performs reductions and arithmetic
4. Linearizer converts result back to string representation

### Key Features
- Capture-avoiding substitution
- Proper precedence handling
- Maximum iteration protection
- Comprehensive error handling
- Type-safe implementation

