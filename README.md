# Lambda Calculus Interpreter in Python

## Installation

Requirements: `python` (`python3`) and `pip` (`pip3`).

Install with `source setup.sh`. Then `python interpreter_test.py` should pass all tests. You can run your own program in `test.lc` with `python interpreter.py test.lc`. 

## Description

The [grammar](https://codeberg.org/alexhkurz/lambdaC-2024/src/branch/main/grammar.lark) supports the standard rules for dropping parentheses with the possible exception of `\a.b \c.d e` which must be written as `\a.b (\c.d e)`. This aligns with standard practice in many functional programming languages and simplifies the grammar. As usual, the following expressions have the same abstract syntax trees:

  - `a b c` = `(a b) c`
  - `\a. \b. c d` = `\a. (\b. c d)`
  
Comments start with `--`.

The workflow followed by the interpreter is defined in [`interpret()`](https://codeberg.org/alexhkurz/lambdaC-2024/src/commit/51a84c820052219a6ce9b7f221cf03db9bd02b0b/interpreter.py#L9-L14).

The interesting functions are [`evaluate()`](https://codeberg.org/alexhkurz/lambdaC-2024/src/commit/483feda11b3f9fbf52f8a5d932e37c0a0560a309/interpreter.py#L37-L50) and [`substitute()`](https://codeberg.org/alexhkurz/lambdaC-2024/src/commit/51a84c820052219a6ce9b7f221cf03db9bd02b0b/interpreter.py#L65-L82).

## Exercises

A [series of exercises](https://hackmd.io/@alexhkurz/S1R1F6_1yx) has been designed to help students explore the material.

