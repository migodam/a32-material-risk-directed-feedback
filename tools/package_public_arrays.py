"""Package the actual authorized online/offline outputs; no hashes or physics."""
from pathlib import Path
import csv,json,re,sys,zipfile
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from src.a32.budget import Book,totals
book=Book(root,'report');out=root/'results/a32';pub=root/'publication';pub.mkdir(exist_ok=True)
gate=json.loads((out/'E0_GATE.json').read_text())
assert gate['status']=='FAIL' and not gate['E1_enabled']
assert totals(root)['new_material_state_evaluations']==0
rows=list(csv.DictReader((out/'offline/E0_CASE_METRICS.csv').open()))
assert len(rows)==608 and len({r['case_id'] for r in rows})==32
assert len({r['parent_id'] for r in rows})==16
assert all(r['success']=='True' for r in rows)
token=json.loads((out/'online/ONLINE_COMPLETE.json').read_text())
files=[out/'online'/f for f in token['prediction_files']]
files += [out/'offline/RECONSTRUCTIONS.npz',out/'offline/DIAGNOSTIC_VECTORS.npz']
assert len(files)==34 and all(f.is_file() for f in files)
# No alternate NPZ, private cache, input tree, or test fixture can enter the ZIP.
with book.span('package_32_online_and_2_offline_arrays',array_files=len(files)):
    archive=pub/'A32_E0_RAW_ARRAYS.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
        for path in files:
            book.check();z.write(path,str(path.relative_to(root)))
        z.writestr('A32_ARRAYS_README.md',
          '# A32 actual E0 arrays\n\nExtract into the repository root.\n'
          'online/:32 saved measurement-conditioned predictions, no target field.\n'
          'offline/RECONSTRUCTIONS.npz: scoring truth and labelled offline capacity.\n'
          'offline/DIAGNOSTIC_VECTORS.npz: cache-only local weak-space diagnosis.\n'
          'No locked data, new labels, native states, or synthetic fixtures.\n'
          'Do not use offline truth-fit arrays as deployable reconstruction inputs.\n')
manifest={'status':'PUBLIC_ARRAYS_PREPARED','archive':archive.name,'bytes':archive.stat().st_size,
          'files':[str(f.relative_to(root)) for f in files],
          'online_cases':32,'independent_exposed_parents':16,
          'new_physical_states':0,'locked_data':False,'hash_checks':False,
          'scientific_status':'E0_FAIL_E1_NOT_RUN',
          'scope':'Actual saved output arrays only; raw sources remain in public A29 Release.'}
(pub/'ARRAY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
# Check current local Markdown destinations, not network resources or file contents.
checked=[]
for doc in ['README.md','A32_E0_E1_DECISION.md','docs/MATHEMATICAL_SCOPE.md']:
    path=root/doc
    for target in re.findall(r'(?<![\w])!?\[[^\]\n]+\]\(([^)]+)\)',path.read_text()):
        if '://' in target or target.startswith('#'):continue
        local=(path.parent/target.split('#')[0]).resolve()
        assert local.exists(),(doc,target)
        checked.append({'document':doc,'target':target})
(pub/'LOCAL_DELIVERY_CHECK.json').write_text(json.dumps({
    'status':'PASS','local_links':checked,'actual_arrays':34,
    'figure_review':'All six actual PNGs inspected; E0 titles and OFFLINE scopes explicit.',
    'scientific_gate':'FAIL','native_E1':'NOT_RUN','hash_checks':False},indent=2)+'\n')
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'actual_arrays':34,
                  'new_physics':0,'local_links':len(checked)}))
