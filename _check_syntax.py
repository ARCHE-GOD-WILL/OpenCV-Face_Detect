import ast, pathlib
for f in sorted(pathlib.Path('opencv_toolkit').rglob('*.py')):
    try:
        ast.parse(f.read_text())
        print(f'{f}: OK')
    except SyntaxError as e:
        print(f'{f}: SYNTAX ERROR - {e}')
for f in sorted(pathlib.Path('examples').rglob('*.py')):
    try:
        ast.parse(f.read_text())
        print(f'{f}: OK')
    except SyntaxError as e:
        print(f'{f}: SYNTAX ERROR - {e}')
