import sys
from lark import Lark, Transformer, Tree
import lark

print(f"Python version: {sys.version}")
print(f"Lark version: {lark.__version__}")

MAX_ITERATIONS = 1000  # Limit for iterations to prevent infinite loops

#  run/execute/interpret source code
def interpret(source_code):
    cst = parser.parse(source_code)
    ast = LambdaCalculusTransformer().transform(cst)
    result_ast = evaluate(ast)
    result = linearize(result_ast)
    return result

# convert concrete syntax to CST
parser = Lark(open("grammar.lark").read(), parser='lalr')

# convert CST to AST
class LambdaCalculusTransformer(Transformer):
    def lam(self, args):
        name, body = args
        return ('lam', str(name), body)

    def app(self, args):
        new_args = [(arg.data, arg.children[0]) if isinstance(arg, Tree) and arg.data == 'int' else arg for arg in args]
        return ('app', *new_args)

    def var(self, args):
        token, = args
        return ('var', str(token))

    def NAME(self, token):
        return str(token)

# reduce AST to normal form
def evaluate(tree):
    iterations = 0
    while iterations < MAX_ITERATIONS:
        if tree[0] == 'app':
            e1 = tree[1]
            e2 = tree[2]
            if e1[0] == 'lam':
                body = e1[2]
                name = e1[1]
                tree = substitute(body, name, e2)
            else:
                e1 = evaluate(e1)
                if e1[0] == 'lam':
                    tree = ('app', e1, e2)
                else:
                    return ('app', e1, evaluate(e2))
        else:
            return tree
        iterations += 1
    return tree  # Return the tree as is if max iterations reached

# generate a fresh name 
# needed eg for \y.x [y/x] --> \z.y where z is a fresh name)
class NameGenerator:
    def __init__(self):
        self.counter = 0

    def generate(self):
        self.counter += 1
        return 'Var' + str(self.counter)

name_generator = NameGenerator()

# for beta reduction (capture-avoiding substitution)
def substitute(tree, name, replacement):
    stack = [(tree, False)]
    result_stack = []
    
    while stack:
        node, visited = stack.pop()
        if visited:
            if node[0] == 'var':
                if node[1] == name:
                    result_stack.append(replacement)
                else:
                    result_stack.append(node)
            elif node[0] == 'lam':
                if node[1] == name:
                    result_stack.append(node)
                else:
                    fresh_name = name_generator.generate()
                    new_body = substitute(substitute(node[2], node[1], ('var', fresh_name)), name, replacement)
                    result_stack.append(('lam', fresh_name, new_body))
            elif node[0] == 'app':
                right = result_stack.pop()
                left = result_stack.pop()
                result_stack.append(('app', left, right))
        else:
            if node[0] == 'app':
                stack.append((node, True))
                stack.append((node[2], False))
                stack.append((node[1], False))
            elif node[0] == 'lam':
                stack.append((node, True))
                stack.append((node[2], False))
            else:
                stack.append((node, True))
    
    if result_stack:
        return result_stack[0]
    else:
        return tree  # Return original tree if no substitution occurred

def linearize(ast):
    if ast[0] == 'var':
        return ast[1]
    elif ast[0] == 'lam':
        return "(" + "\\" + ast[1] + "." + linearize(ast[2]) + ")"
    elif ast[0] == 'app':
        return "(" + linearize(ast[1]) + " " + linearize(ast[2]) + ")"
    else:
        raise Exception('Unknown AST', ast)

def main():
    if len(sys.argv) != 2:
        print("Usage: python interpreter.py <filename>", file=sys.stderr)
        sys.exit(1)

    filename = sys.argv[1]
    with open(filename, 'r') as file:
        expressions = file.read().split('\n')

    for expression in expressions:
        if expression.strip() and not expression.strip().startswith('--'):
            result = interpret(expression)
            print(f"Expression: {expression}")
            print(f"Result: \033[95m{result}\033[0m")
            print()

if __name__ == "__main__":
    main()
