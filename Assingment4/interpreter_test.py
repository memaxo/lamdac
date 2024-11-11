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

if __name__ == "__main__":
    unittest.main()
