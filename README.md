# Lambda Calculus Interpreter with Arithmetic

A sophisticated implementation of a lambda calculus interpreter extended with arithmetic operations. This project demonstrates fundamental concepts in programming language theory including parsing, evaluation strategies, and type-safe interpretation.

## Features

- Full lambda calculus implementation
- Arithmetic operations (+, -, *, /)
- Lazy evaluation strategy
- Capture-avoiding substitution
- Proper operator precedence
- Comprehensive error handling

## Quick Start

1. Requirements:
   - Python 3.8 or higher
   - pip package manager

2. Installation:
   ```
   source setup.sh
   ```

3. Run tests:
   ```
   python interpreter_test.py
   ```

4. Run your program:
   ```
   python interpreter.py your_program.lc
   ```

## Language Guide

### Basic Syntax

1. Lambda Expressions:
   - Abstraction: `\x.e`
   - Application: `f x`
   - Variables: `x`, `y`, `z`

2. Arithmetic:
   - Addition: `x + y`
   - Subtraction: `x - y`
   - Multiplication: `x * y`
   - Division: `x / y`
   - Negation: `-x`

3. Numbers:
   - Integer literals: `1`, `2`, `3`
   - Negative numbers: `-1`, `-2`
   - Floating point: `1.5`, `2.0`

### Operator Precedence (highest to lowest)

1. Parentheses `()`
2. Function application
3. Negation `-`
4. Multiplication `*` and division `/`
5. Addition `+` and subtraction `-`
6. Lambda abstraction `\`

### Examples

1. Basic Lambda Calculus:
   ```
   (\x.x) y              => y
   (\x.\y.x) a b         => a
   (\x.x x) (\x.x)       => (\x.x)
   ```

2. Arithmetic:
   ```
   1 + 2 * 3             => 7
   (\x.x * x) 3          => 9
   (\x.\y.x + y) 3 4     => 7
   ```

3. Mixed Expressions:
   ```
   (\x.x + 1) 5          => 6
   (\x.x * x + 2) 3      => 11
   ```

## Implementation Details

### Core Components

1. Parser (`grammar.lark`):
   - Defines language syntax using Lark grammar
   - Handles operator precedence and associativity
   - Manages whitespace and comments

2. AST (`interpreter.py`):
   - Type-safe expression representation
   - Dataclasses for each expression type
   - Clear separation of concerns

3. Evaluator:
   - Lazy evaluation strategy
   - Capture-avoiding substitution
   - Arithmetic evaluation
   - Error handling

### Error Handling

1. Parser Errors:
   - Syntax errors
   - Invalid tokens
   - Unexpected input

2. Evaluation Errors:
   - Division by zero
   - Arithmetic overflow
   - Maximum recursion depth
   - Type mismatches

3. Runtime Protection:
   - Iteration limits
   - Stack overflow prevention
   - Memory protection

## Testing

The test suite (`interpreter_test.py`) covers:

1. Core Lambda Calculus:
   - Beta reduction
   - Variable capture
   - Complex applications

2. Arithmetic Operations:
   - Basic arithmetic
   - Operator precedence
   - Mixed expressions

3. Error Cases:
   - Division by zero
   - Overflow conditions
   - Invalid syntax

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

Based on the lambda calculus implementation guidelines from Programming Languages course at Chapman University.

