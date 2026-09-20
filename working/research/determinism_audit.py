"""AST audit against the six categories the ERIS Deterministic Execution check names:
fallback, training, inference, worker, seed, backend."""
import ast, sys

BANNED_CALLS = {'time', 'perf_counter', 'monotonic', 'is_available', 'device_count',
                'mem_get_info', 'get_device_properties', 'getenv', 'cpu_count',
                'is_bf16_supported', 'current_device', 'memory_allocated'}
path = sys.argv[1]
src = open(path, encoding='utf-8').read()
tree = ast.parse(src)
bad = []
for node in ast.walk(tree):
    if isinstance(node, ast.Try):
        bad.append(f'L{node.lineno}: try/except')
    if isinstance(node, ast.While):
        bad.append(f'L{node.lineno}: while loop')
    if isinstance(node, ast.Call):
        f = node.func
        nm = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else '')
        if nm in BANNED_CALLS:
            bad.append(f'L{node.lineno}: call {nm}()')
        if nm == 'setdefault' and isinstance(f.value, ast.Attribute) and f.value.attr == 'environ':
            bad.append(f'L{node.lineno}: os.environ.setdefault')
    if isinstance(node, ast.Import):
        for a in node.names:
            if a.name in ('time', 'random', 'multiprocessing'):
                bad.append(f'L{node.lineno}: import {a.name}')
    if isinstance(node, ast.ImportFrom) and node.module in ('time', 'random'):
        bad.append(f'L{node.lineno}: from {node.module} import')

print(f'--- {path} ---')
if bad:
    print('VIOLATIONS:')
    for b in bad:
        print('  ' + b)
else:
    print('clean: no try/while/clock/device-probe/setdefault constructs')

ifs = [n for n in ast.walk(tree) if isinstance(n, ast.If)]
print(f'{len(ifs)} `if` statements to eyeball:')
for n in ifs:
    print(f'  L{n.lineno}: {ast.unparse(n.test)[:90]}')
sys.exit(1 if bad else 0)
