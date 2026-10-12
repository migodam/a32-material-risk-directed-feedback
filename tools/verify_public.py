"""Anonymous web availability and Git commit identity; no file hashing."""
from pathlib import Path
import json,subprocess,sys,time,urllib.request
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from src.a32.budget import Book,totals
book=Book(root,'report');pub=root/'publication';pub.mkdir(exist_ok=True)
repo='migodam/a32-material-risk-directed-feedback'
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
headers={'User-Agent':'A32-public-evidence-check','Accept':'application/vnd.github+json'}
def get(url,method='GET',read=True):
    book.check()
    req=urllib.request.Request(url,method=method,headers=headers)
    with urllib.request.urlopen(req,timeout=30) as r:
        row={'url':url,'HTTP_status':r.status,'anonymous':True,
             'content_type':r.headers.get('Content-Type'),
             'content_length_header':r.headers.get('Content-Length')}
        body=r.read() if read else None
        assert r.status==200
    return row,body
api,body=get('https://api.github.com/repos/'+repo)
meta=json.loads(body);assert meta['private'] is False and meta['default_branch']=='main'
commit_row,body=get('https://api.github.com/repos/'+repo+'/commits/main')
remote_sha=json.loads(body)['sha'];assert sha==remote_sha,(sha,remote_sha)
links=[api,commit_row]
for url,needle in [
 ('https://github.com/'+repo,'A32'),
 ('https://github.com/'+repo+'/blob/'+sha+'/A32_E0_E1_DECISION.md','A32_E0_E1_DECISION'),
 ('https://raw.githubusercontent.com/'+repo+'/'+sha+'/README.md','E0 FAIL'),
 ('https://raw.githubusercontent.com/'+repo+'/'+sha+'/A32_E0_E1_DECISION.md','39.46%'),
 ('https://github.com/'+repo+'/releases/tag/a32-e0-v0.1.0','a32-e0-v0.1.0')]:
    row,body=get(url);assert needle in body.decode('utf-8');links.append(row)
row,body=get('https://api.github.com/repos/'+repo+'/releases/tags/a32-e0-v0.1.0')
release=json.loads(body);links.append(row)
asset=next(a for a in release['assets'] if a['name']=='A32_E0_RAW_ARRAYS.zip')
expected=json.loads((pub/'ARRAY_MANIFEST.json').read_text())['bytes']
assert asset['size']==expected
asset_row,_=get(asset['browser_download_url'],method='HEAD',read=False);links.append(asset_row)
receipt={'status':'VERIFIED_PUBLIC','repository_url':'https://github.com/'+repo,
         'report_url':'https://github.com/'+repo+'/blob/'+sha+'/A32_E0_E1_DECISION.md',
         'raw_report_url':'https://raw.githubusercontent.com/'+repo+'/'+sha+'/A32_E0_E1_DECISION.md',
         'release_url':release['html_url'],'actual_commit':sha,
         'scientific_code_release_commit':'f425484777a1da5fe625148772314ad3c30b00b9',
         'asset':{'url':asset['browser_download_url'],'bytes':asset['size'],'anonymous_HTTP':asset_row['HTTP_status']},
         'checks':links,'scientific_status':'E0_FAIL_E1_NOT_RUN',
         'physical_accounting_at_check':totals(root),
         'hash_checks':False,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(pub/'PUBLICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':'VERIFIED_PUBLIC','commit':sha,'asset_bytes':asset['size'],
                  'repository':receipt['repository_url'],'report':receipt['report_url']}))
