"""Audit final template packages and index the completed case artifacts.

No experiments or archive mutations. Existing final archive hash audits are
retained; only postprocessing code, report and document hashes are refreshed.
"""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import hashlib, json, datetime, re, os

r = Path(__file__).resolve().parents[1]
docs = r / 'documents'
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()
def entry(p): return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}
def xml(z, name): return E.fromstring(z.read(name))
def canonical(nodes): return [E.tostring(n, method='c14n') for n in nodes]

audits = {}
for kind, filename, figure_count in [('design', 'case-executed-design.docx', 0),
                                      ('analysis', 'case-analysis.docx', 3)]:
    qa = docs / 'qa' / kind
    contract = json.loads((qa / 'template-contract.json').read_text())
    ref = Path(contract['reference']); final = docs / filename
    assert sha(ref) == contract['sha256']
    with ZipFile(ref) as a, ZipFile(final) as b:
        allowed = {'word/document.xml', 'word/settings.xml'}
        allowed.update(n for n in a.namelist() if n.startswith(('word/header', 'word/footer')))
        if figure_count: allowed |= {'word/_rels/document.xml.rels', '[Content_Types].xml'}
        for name in a.namelist():
            if name not in allowed: assert a.read(name) == b.read(name), name
        ar = xml(a, 'word/document.xml'); br = xml(b, 'word/document.xml')
        for xpath in ['//w:sectPr', '//w:tblPr', '//w:tblGrid', '//w:trPr', '//w:tcPr']:
            assert canonical(ar.xpath(xpath, namespaces=ns)) == canonical(br.xpath(xpath, namespaces=ns)), xpath
        assert len(ar.xpath('//w:sdt', namespaces=ns)) == len(br.xpath('//w:sdt', namespaces=ns))
        assert ar.xpath('//w:instrText/text()', namespaces=ns) == br.xpath('//w:instrText/text()', namespaces=ns)
        assert len(br.xpath('//w:drawing', namespaces=ns)) == len(ar.xpath('//w:drawing', namespaces=ns)) + figure_count
        oldrels = {x.get('Id'): dict(x.attrib) for x in xml(a, 'word/_rels/document.xml.rels')}
        newrels = {x.get('Id'): dict(x.attrib) for x in xml(b, 'word/_rels/document.xml.rels')}
        assert all(newrels[k] == v for k, v in oldrels.items())
        assert len(newrels) - len(oldrels) == figure_count
        added = set(b.namelist()) - set(a.namelist())
        assert len(added) == figure_count and all(x.startswith('word/media/case-memory-') for x in added)
        changed = [n for n in a.namelist() if a.read(n) != b.read(n)]
    result = {'reference_unchanged': True, 'preserve_only_parts_exact': True,
              'section_geometry_unchanged': True, 'table_geometry_unchanged': True,
              'controls_and_fields_preserved': True, 'existing_relationships_preserved': True,
              'added_figures': figure_count, 'changed_parts': changed,
              'intentional_adaptations': contract['intentional_adaptations'] +
              (['Three requested images; keepNext on headings, intervening spacers and table headers.'] if figure_count else []),
              'reference_sha256': sha(ref), 'output_sha256': sha(final)}
    (qa / 'structural-qa.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    audits[kind] = result

analysis_render = docs / 'qa/analysis/fidelity-final-delivery/b_render'
previous = docs / 'qa/analysis/fidelity-publish/b_render'
analysis_pages = []
for i in range(1, 13):
    p = analysis_render / f'page-{i}.png'
    unchanged = sha(p) == sha(previous / p.name)
    assert unchanged if i not in [5, 6, 7, 8, 9] else not unchanged
    analysis_pages.append({'page': i, 'render': str(p), 'sha256': sha(p),
                           'inspection': 'Final page visually inspected' if not unchanged else
                           'Pixel-identical to visually inspected prior page'})
design_render = docs / 'qa/design/fidelity-final/b_render'
design_pages = [{'page': i, 'render': str(design_render / f'page-{i}.png'),
                 'sha256': sha(design_render / f'page-{i}.png'),
                 'inspection': 'Final page visually inspected'} for i in range(1, 7)]
completion = {'complete': True, 'documents': 2, 'all_pages_reviewed': True,
              'design_pages': design_pages, 'analysis_pages': analysis_pages,
              'structural_checks': audits,
              'visual_result': 'Readable CJK, complete tables, headings attached to content, three complete figure pages; no cropping or overlap.',
              'scope': 'Delivery QA; does not replace experimental correctness audits.'}
(docs / 'qa/completion.json').write_text(json.dumps(completion, ensure_ascii=False, indent=2) + '\n')

report = r.parent / 'case-report.md'
for target in re.findall(r'\]\((/[^\n)]+)\)', report.read_text()):
    assert Path(target).exists(), target
assert json.loads((r / 'statistics.json').read_text())['complete']
assert json.loads((r / 'correctness-audit.json').read_text())['complete']
assert json.loads((r / 'address-diagnosis-verification.json').read_text())['passed']

provenance_path = r / 'final-provenance.json'
provenance = json.loads(provenance_path.read_text())
assert provenance['formal_processes'] == 28 and len(provenance['local_archive_final_hash_audit']) == 8
for p in sorted((r / 'code').iterdir()):
    if p.is_file() and p.suffix != '.pyc':
        provenance['local_code'][str(p.relative_to(r))] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
remote = json.loads((r / 'remote-provenance.json').read_text())
for name in provenance['critical_local_remote_sources_identical']:
    assert sha(r / name) == remote['case_files'][name]['sha256']
delivered = [report, r.parent / 'protocol-executed.md', r.parent / 'README.md',
             r / 'address-diagnosis-verification.json', docs / 'case-analysis.docx',
             docs / 'case-executed-design.docx', docs / 'qa/completion.json']
for p in delivered:
    provenance['evidence'][os.path.relpath(p, r)] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
provenance['delivery_updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
provenance['delivery_note'] = 'Postprocessing/document hashes refreshed. Prior final full-archive SHA audits retained; timed producer/analyzer and fine probe sources remain identical to actual remote runtime inventory.'
provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
index = {'complete': True, 'formal_processes': 28, 'paired_blocks': 7,
         'artifacts': [entry(p) for p in delivered + [provenance_path]],
         'figures': [entry(r / f'figures/memory-layout-{token}.{ext}') for token in [0,1,63] for ext in ['png','pdf']],
         'raw_evidence_index': 'final-provenance.json and capture/rN-condition/retention.json',
         'frozen_evidence_untouched': provenance['frozen_evidence_untouched']}
(r / 'delivery-index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'complete': True, 'requests': 28, 'documents': 2, 'reviewed_pages': 18, 'figures': 3}))
