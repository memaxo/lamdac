# Lambda Calculus Interpreter in Python

## Installation

Install with `source setup.sh`. Then `python interpreter_test.py` should pass all tests. You can run your own program in `test.lc` with `python interpreter.py test.lc`. Comments start with `--`.

## Description

Supports the standard rules for dropping parentheses with the possible exception of `\a.b \c.d e` which must be written as `\a.b (\c.d e)`. This aligns with standard practice in many functional programming languages and simplifies the grammar. As usual, the following expressions have the same abstract syntax trees:
  - `a b c` = `(a b) c`
  - `\a. \b. c d` = `\a. (\b. c d)`

## Exercises

A [series of exercises](https://hackmd.io/@alexhkurz/S1R1F6_1yx) has been designed to help students explore the material.

