import unittest
from interpreter import (
    interpret,
    substitute,
    evaluate,
    LambdaCalculusTransformer,
    parser,
    linearize,
    Var,
    Lam,
    App,
    Num,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    EvaluationError,
    ParserError,
)
import lark

def ast(source_code):
    """Parses the source code into an abstract syntax tree (AST)."""
    return LambdaCalculusTransformer().transform(parser.parse(source_code))

class TestParsing(unittest.TestCase):
    """Tests for parsing expressions into the correct AST."""

    def test_basic_lambda(self):
        self.assertEqual(ast("\\x.y"), Lam('x', Var('y')))
        self.assertEqual(ast("\\x.x"), Lam('x', Var('x')))
        self.assertEqual(ast("\\x.\\y.x"), Lam('x', Lam('y', Var('x'))))

    def test_lambda_with_application(self):
        self.assertEqual(ast("\\x.x y"), Lam('x', App(Var('x'), Var('y'))))
        self.assertEqual(ast("(\\x.x) y"), App(Lam('x', Var('x')), Var('y')))
        self.assertEqual(ast("\\x.\\y.x y"), Lam('x', Lam('y', App(Var('x'), Var('y')))))

    def test_arithmetic(self):
        self.assertEqual(ast("1 + 2"), Add(Num(1.0), Num(2.0)))
        self.assertEqual(ast("1 * 2"), Mul(Num(1.0), Num(2.0)))
        self.assertEqual(ast("-1"), Num(-1.0))  # Updated: Negative numbers are handled directly
        self.assertEqual(ast("1 - 2"), Sub(Num(1.0), Num(2.0)))
        self.assertEqual(ast("1 * 2 + 3"), Add(Mul(Num(1.0), Num(2.0)), Num(3.0)))
        self.assertEqual(ast("6 / 2"), Div(Num(6.0), Num(2.0)))

    def test_mixed_expressions(self):
        self.assertEqual(ast("\\x.x * x"), Lam('x', Mul(Var('x'), Var('x'))))
        self.assertEqual(ast("(\\x.x) 1 + 2"), Add(App(Lam('x', Var('x')), Num(1.0)), Num(2.0)))
        self.assertEqual(ast("\\x.x + 1 * 2"), Lam('x', Add(Var('x'), Mul(Num(1.0), Num(2.0)))))
        self.assertEqual(ast("(\\x.x * x) (-2)"), App(Lam('x', Mul(Var('x'), Var('x'))), Num(-2.0)))  # Updated

class TestSubstitution(unittest.TestCase):
    """Tests for the substitution function."""

    def test_basic_substitution(self):
        expr = Var('x')
        result = substitute(expr, 'x', Var('y'))
        expected = Var('y')
        self.assertEqual(result, expected)

    def test_substitution_with_lambda(self):
        expr = Lam('x', Var('x'))
        result = substitute(expr, 'x', Var('y'))
        expected = expr  # Should remain unchanged
        self.assertEqual(result, expected)

    def test_substitution_in_application(self):
        expr = App(Var('x'), Var('x'))
        result = substitute(expr, 'x', Var('y'))
        expected = App(Var('y'), Var('y'))
        self.assertEqual(result, expected)

    def test_variable_capture_avoidance(self):
        expr = Lam('y', Var('x'))
        result = substitute(expr, 'x', Var('y'))
        # Since we cannot rely on specific variable names, check the structure instead
        self.assertIsInstance(result, Lam)
        self.assertNotEqual(result.param, 'y')  # Parameter should be renamed
        self.assertEqual(result.body, Var('y'))  # Body should have the substituted variable

    def test_arithmetic_substitution(self):
        expr = Add(Var('x'), Num(1.0))
        result = substitute(expr, 'x', Num(5.0))
        expected = Add(Num(5.0), Num(1.0))
        self.assertEqual(result, expected)

        expr = Mul(Var('x'), Var('x'))
        result = substitute(expr, 'x', Num(2.0))
        expected = Mul(Num(2.0), Num(2.0))
        self.assertEqual(result, expected)

        expr = Neg(Var('x'))
        result = substitute(expr, 'x', Num(3.0))
        expected = Neg(Num(3.0))
        self.assertEqual(result, expected)

        expr = Div(Var('x'), Num(2.0))
        result = substitute(expr, 'x', Num(6.0))
        expected = Div(Num(6.0), Num(2.0))
        self.assertEqual(result, expected)

class TestEvaluation(unittest.TestCase):
    """Tests for evaluating expressions."""

    def test_lambda_calculus_evaluation(self):
        self.assertEqual(linearize(evaluate(ast("\\x.x"))), "(\\x.x)")
        self.assertEqual(linearize(evaluate(ast("(\\x.x) y"))), "y")
        self.assertEqual(linearize(evaluate(ast("(\\x.x x) y"))), "(y y)")

    def test_arithmetic_evaluation(self):
        self.assertEqual(linearize(evaluate(ast("1 + 2"))), "3.0")
        self.assertEqual(linearize(evaluate(ast("2 * 3"))), "6.0")
        self.assertEqual(linearize(evaluate(ast("-2"))), "-2.0")
        self.assertEqual(linearize(evaluate(ast("1 - 2 * 3"))), "-5.0")
        self.assertEqual(linearize(evaluate(ast("6 / 2"))), "3.0")

    def test_mixed_evaluation(self):
        self.assertEqual(linearize(evaluate(ast("(\\x.x + 1) 2"))), "3.0")
        self.assertEqual(linearize(evaluate(ast("(\\x.x * x) 3"))), "9.0")
        self.assertEqual(linearize(evaluate(ast("(\\x.x) (1 + 2)"))), "3.0")

    def test_division_by_zero(self):
        with self.assertRaises(EvaluationError):
            evaluate(ast("1 / 0"))

class TestPrecedence(unittest.TestCase):
    """Tests for operator precedence."""

    def test_arithmetic_precedence(self):
        self.assertEqual(interpret("1 + 2 * 3"), "7.0")
        self.assertEqual(interpret("1 * 2 + 3"), "5.0")
        self.assertEqual(interpret("-1 * 2"), "-2.0")
        self.assertEqual(interpret("1 - 2 * 3"), "-5.0")
        self.assertEqual(interpret("(1 + 2) * 3"), "9.0")

    def test_lambda_and_arithmetic_precedence(self):
        self.assertEqual(interpret("\\x.x * x"), "(\\x.(x * x))")
        self.assertEqual(interpret("(\\x.x) 1 + 2"), "3.0")
        self.assertEqual(interpret("\\x.x + 1 * 2"), "(\\x.(x + (1.0 * 2.0)))")
        self.assertEqual(interpret("(\\x.x * x) 2 * 3"), "12.0")
        self.assertEqual(interpret("(\\x.x * x) (-2) * (-3)"), "-12.0")

    def test_complex_expressions(self):
        # ((λx.x * x) -2) * -3 = 4 * -3 = -12
        self.assertEqual(interpret("((\\x.x * x) -2) * -3"), "-12.0")
        
        # (λx.x * x) (-2) * (-3) = 4 * -3 = -12
        self.assertEqual(interpret("(\\x.x * x) (-2) * (-3)"), "-12.0")

class TestErrorHandling(unittest.TestCase):
    """Tests for error handling scenarios."""
    
    def test_division_by_zero(self):
        with self.assertRaises(EvaluationError):
            interpret("1 / 0")
        with self.assertRaises(EvaluationError):
            interpret("1 / (1 - 1)")
            
    def test_invalid_syntax(self):
        with self.assertRaises(ParserError):
            interpret("\\")
        with self.assertRaises(ParserError):
            interpret("1 + ")
            
    def test_empty_input(self):
        with self.assertRaises(ParserError):
            interpret("")
        with self.assertRaises(ParserError):
            interpret("   ")
            
    def test_overflow_protection(self):
        with self.assertRaises(EvaluationError):
            interpret("1e308 + 1e308")  # Adjusted to a realistic overflow scenario

class TestAdvancedFeatures(unittest.TestCase):
    """Tests for advanced language features."""
    
    def test_nested_arithmetic(self):
        self.assertEqual(interpret("1 + (2 * (3 + 4))"), "15.0")
        self.assertEqual(interpret("((1 + 2) * 3) / 2"), "4.5")
        
    def test_complex_lambda_expressions(self):
        self.assertEqual(
            interpret("(\\x.\\y.x + y) 1 2"), 
            "3.0"
        )
        self.assertEqual(
            interpret("(\\x.\\y.\\z.x * y + z) 2 3 4"),
            "10.0"
        )
        
    def test_lambda_with_arithmetic(self):
        self.assertEqual(
            interpret("(\\x.x * x + 2 * x + 1) 3"),
            "16.0"
        )

class TestMilestone2Features(unittest.TestCase):
    """Tests for Milestone 2 language features."""
    
    def test_conditional_expressions(self):
        # Basic if-then-else
        self.assertEqual(interpret("if 1 <= 2 then 3 else 4"), "3.0")
        self.assertEqual(interpret("if 2 <= 1 then 3 else 4"), "4.0")
        
        # Additional basic cases
        self.assertEqual(interpret("if 0 then 2 else 1"), "1.0")
        self.assertEqual(interpret("if 1 then 2 else 2"), "2.0")
        
        # Nested conditionals
        self.assertEqual(
            interpret("if 1 <= 2 then if 3 <= 4 then 5 else 6 else 7"),
            "5.0"
        )
        self.assertEqual(interpret("if 0 then 2 else if 1 then 3 else 4"), "3.0")
        self.assertEqual(interpret("if 0 then 2 else if 0 then 3 else 4"), "4.0")
        self.assertEqual(interpret("if 1 <= 2 then if 0 then 3 else 4 else 5"), "4.0")
        
        # Conditionals with arithmetic
        self.assertEqual(
            interpret("if 1 + 2 <= 4 then 5 * 2 else 6 / 2"),
            "10.0"
        )

    def test_comparison_operators(self):
        # Less than or equal
        self.assertEqual(interpret("1 <= 2"), "1.0")  # true
        self.assertEqual(interpret("2 <= 1"), "0.0")  # false
        self.assertEqual(interpret("2 <= 2"), "1.0")  # true
        self.assertEqual(interpret("1 <= 1"), "1.0")  # true
        
        # Equality
        self.assertEqual(interpret("1 == 1"), "1.0")  # true
        self.assertEqual(interpret("1 == 2"), "0.0")  # false
        self.assertEqual(interpret("0 == 0"), "1.0")  # true
        
        # Complex comparisons
        self.assertEqual(interpret("1 + 2 == 3"), "1.0")
        self.assertEqual(interpret("2 * 3 <= 7"), "1.0")
        
    def test_let_expressions(self):
        # Basic let
        self.assertEqual(interpret("let x = 5 in x"), "5.0")
        self.assertEqual(interpret("let x = 5 in x + 3"), "8.0")
        
        # Nested let
        self.assertEqual(
            interpret("let x = 5 in let y = 3 in x + y"),
            "8.0"
        )
        self.assertEqual(
            interpret("let x = 2 in let y = x + 3 in y * 2"),
            "10.0"
        )
        
        # Let with lambda
        self.assertEqual(
            interpret("let f = \\x.x * x in f 3"),
            "9.0"
        )
        
        # Nested let with shadowing
        self.assertEqual(
            interpret("let x = 1 in let x = x + 1 in x"),
            "2.0"
        )

    def test_function_composition(self):
        # Function composition tests
        self.assertEqual(
            interpret("let f = \\x.x + 1 in let g = \\x.x * 2 in g (f 3)"),
            "8.0"
        )
        self.assertEqual(
            interpret("let f = \\x.x + 1 in let g = \\x.x * 2 in f (g 3)"),
            "7.0"
        )

    def test_error_handling(self):
        # Invalid expressions should raise appropriate errors
        with self.assertRaises(ParserError):
            interpret("if 1 then else 2")
        with self.assertRaises(ParserError):
            interpret("let = 1 in 2")
        with self.assertRaises(EvaluationError):
            interpret("fix 1")

    def test_letrec_expressions(self):
        # Basic recursive function
        self.assertEqual(
            interpret(
                "letrec fact = \\n.if n <= 1 then 1 else n * fact (n - 1) in fact 3"
            ),
            "6.0"
        )
        
        # Mutual recursion
        self.assertEqual(
            interpret(
                "letrec even = \\n.if n == 0 then 1 else if n == 1 then 0 else even (n - 2) in even 4"
            ),
            "1.0"
        )

    def test_fix_operator(self):
        # Factorial using fix
        self.assertEqual(
            interpret(
                "let factorial = fix \\f.\\n.if n <= 1 then 1 else n * f (n - 1) in factorial 3"
            ),
            "6.0"
        )
        
        # Fibonacci using fix
        self.assertEqual(
            interpret(
                "let fib = fix \\f.\\n.if n <= 1 then n else f (n - 1) + f (n - 2) in fib 5"
            ),
            "5.0"
        )

    def test_complex_milestone2_expressions(self):
        # Combination of features
        self.assertEqual(
            interpret(
                "let max = \\x.\\y.if x <= y then y else x in max 3 4"
            ),
            "4.0"
        )
        
        # Recursive function with comparison and arithmetic
        self.assertEqual(
            interpret(
                "letrec sum = \\n.if n <= 0 then 0 else n + sum (n - 1) in sum 5"
            ),
            "15.0"
        )
        
        # Complex conditional with let and arithmetic
        self.assertEqual(
            interpret(
                "let x = 5 in let y = 3 in if x <= y then x * y else x + y"
            ),
            "8.0"
        )

if __name__ == "__main__":
    unittest.main()
