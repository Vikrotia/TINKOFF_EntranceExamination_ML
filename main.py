import ast
import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('input', type=str, help='input')
parser.add_argument('score', type=str, help='score')
args = parser.parse_args()


class CommandLineArguments(object):
    def __init__(self):
        self.filepath_input = args.input
        self.filepath_score = args.score
        self.files = []

    def readFileInput(self):
        with open(self.filepath_input, 'r') as f:
            for line in f:
                self.files.append(line)
            for i in range(len(self.files)):
                self.files[i] = self.files[i].split()

    def writingToFile(self, other):
        with open(self.filepath_score, 'a') as f:
            f.write(other)


class AstProcessing(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.code_str = ''
        self.ast_code_tree = ''
        self.str_of_dump_tree = ''

    def readFile(self):
        with open(self.filepath, 'r') as f:
            self.code_str = f.read()

    def createAst(self):
        self.ast_code_tree = ast.parse(self.code_str)

    def removeDocstring(self):
        l = []
        for f in ast.walk(self.ast_code_tree):
            if isinstance(f, ast.FunctionDef):
                if ast.get_docstring(f) is not None:
                    l.append('"""' + ast.get_docstring(f) + '"""')
        for i in l:
            self.code_str = self.code_str.replace(i, '')

    def getLower(self):
        self.code_str = self.code_str.lower()

    def LevenshteinDistance(a, b):
        n, m = len(a), len(b)
        if n > m:
            a, b = b, a
            n, m = m, n
        current_row = range(n + 1)
        for i in range(1, m + 1):
            previous_row, current_row = current_row, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete, change = previous_row[j] + \
                    1, current_row[j - 1] + 1, previous_row[j - 1]
                if a[j - 1] != b[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)
        return current_row[n]

    def dumpCode(self):
        self.str_of_dump_tree = ast.dump(self.ast_code_tree)

    def processing(self):
        self.readFile()
        self.createAst()
        self.removeDocstring()
        self.createAst()
        self.getLower()
        self.dumpCode()


m = CommandLineArguments()
m.readFileInput()
for i in range(len(m.files)):
    m1 = AstProcessing(m.files[i][0])
    m2 = AstProcessing(m.files[i][1])
    m1.processing()
    m2.processing()
    m.writingToFile(str(round(AstProcessing.LevenshteinDistance(m1.str_of_dump_tree,
            m2.str_of_dump_tree) / len(m2.str_of_dump_tree), 3)) + '\n')
