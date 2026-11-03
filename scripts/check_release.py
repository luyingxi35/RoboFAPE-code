"""Static release checks; passing does not imply GPU runtime validation."""
import ast
import builtins
import json
from pathlib import Path
import re
import symtable

root=Path(__file__).resolve().parents[1]
errors=[]
for path in (root/'src').rglob('*.py'):
    content=path.read_text(); ast.parse(content,filename=str(path))
    table=symtable.symtable(content,str(path),'exec')
    defined={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    def check(scope):
        for symbol in scope.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in defined|set(dir(builtins))|{'__name__','__file__','__package__'}:
                errors.append(f'{path.relative_to(root)}: unresolved global {symbol.get_name()}')
        for child in scope.get_children():check(child)
    check(table)
    if 'sys.path.' in content: errors.append(f'{path.relative_to(root)}: sys.path mutation')
    if re.search(r'/(?:data|home|Users)/[A-Za-z]',content): errors.append(f'{path.relative_to(root)}: private absolute path')
for path in root.rglob('*.md'):
    if any(x in path.parts for x in ['build','dist','outputs','.venv']):continue
    for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
        if '://' in link or link.startswith('#'):continue
        target=(path.parent/link.split('#')[0]).resolve()
        if not target.is_file(): errors.append(f'{path.relative_to(root)}: broken link {link}')
for path in (root/'src/robofape/configs').glob('*.json'):
    data=json.loads(path.read_text())
    module=data['runtime']['task_module']
    if module and not (root/f'src/robofape/sim/tasks/task_{module}.py').is_file():errors.append(f'{path.name}: task module missing')
    module,name=data['runtime']['solver'].split(':',1)
    if module!='upstream':
        solver=root/f'src/robofape/sim/solvers/solve_{module}.py'
        if not solver.is_file() or name not in {n.name for n in ast.parse(solver.read_text()).body if isinstance(n,ast.FunctionDef)}:
            errors.append(f'{path.name}: solver missing')
required=['LICENSE','NOTICE.md','README.md','docs/reproduction.md','docs/policy.md','docs/tasks.md','docs/release_checklist.md','docs/assets/overview.jpeg']
for name in required:
    if not (root/name).is_file():errors.append(f'missing {name}')
if errors:raise SystemExit('\n'.join(errors))
print('Static release checks: PASS (GPU/policy/sample checks are separate)')
