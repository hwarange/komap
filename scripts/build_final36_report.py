"""Build the revised train70 report (v2) by filling the retained contest form.

Differences from build_final35_report.py: content flows in consecutive outer rows without
forced page breaks, figures are embedded inside the form, every table carries a numbered
caption, and the output is a new file so the earlier submission file is preserved.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
EXTRA = Path(r'C:/Users/SSAFY/AppData/Local/Temp/claude/c--Users-SSAFY-Desktop-Komap/bc946a9d-e605-42e6-80fc-d5ff55c5d015/scratchpad/pylib')
if EXTRA.exists():
    sys.path.insert(0, str(EXTRA))
from docx import Document  # noqa: E402
from docx.shared import Inches  # noqa: E402
from lxml import etree as E  # noqa: E402
from PIL import Image  # noqa: E402

WORK = ROOT / 'reports/final36_document_work'
REFERENCE = ROOT / 'reports/modeling_document_work/reference.docx'
OUT = ROOT / 'reports/모델링결과서_김형수없는김형수_최종제출v2.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
NS = {'w': W, 'r': R}
LINE = 264      # 1.1 line spacing (240 = single)
AFTER = 60      # paragraph spacing after, twips/20


def node(tag, **attrs):
    x = E.Element('{' + W + '}' + tag)
    for k, v in attrs.items():
        x.set('{' + W + '}' + k, str(v))
    return x


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Form:
    def __init__(self):
        assert sha(REFERENCE) == 'a9f66f14f8546b37d6a3a4c2c7406c328d95bb6f7b56b501abce9e4bfba0b534'
        with ZipFile(REFERENCE) as z:
            self.parts = {n: z.read(n) for n in z.namelist()}
            self.infos = z.infolist()
        self.xml = E.fromstring(self.parts['word/document.xml'])
        self.section = E.tostring(self.xml.find('.//w:sectPr', NS))
        self.rows = self.xml.find('.//w:tbl', NS).findall('w:tr', NS)
        self.body = self.rows[9].find('w:tc', NS)
        p = self.body.findall('w:p', NS)
        self.samples = {'body': p[7], 'heading': p[3], 'sub': p[5]}
        self.rels = E.fromstring(self.parts['word/_rels/document.xml.rels'])
        self.image_count = 0
        self.table_count = 0
        self.texts = []
        self.images = []
        self.cover_mode = False
        self.clear(self.body)
        rp = self.rows[9].find('w:trPr', NS)
        if rp is not None:
            for x in list(rp):
                if x.tag in ['{' + W + '}trHeight', '{' + W + '}cantSplit']:
                    rp.remove(x)
        v = self.body.find('w:tcPr/w:vAlign', NS)
        if v is not None:
            v.set('{' + W + '}val', 'top')

    def clear(self, cell):
        for x in list(cell):
            if x.tag != '{' + W + '}tcPr':
                cell.remove(x)

    def paragraph(self, text, role='body', size=None, keep=False, center=False, bold=None):
        p = deepcopy(self.samples.get(role, self.samples['body']))
        for x in list(p):
            if x.tag != '{' + W + '}pPr':
                p.remove(x)
        pp = p.find('w:pPr', NS)
        if pp is None:
            pp = node('pPr'); p.insert(0, pp)
        for name in ['ind', 'spacing', 'keepNext', 'keepLines', 'widowControl', 'jc']:
            for x in pp.findall('w:' + name, NS):
                pp.remove(x)
        before = 140 if role == 'heading' else (100 if role == 'sub' else 0)
        after = {'caption': 40, 'ref': 30}.get(role, AFTER)
        pp.append(node('spacing', before=before, after=after,
                       line=300 if self.cover_mode else (240 if role == 'ref' else LINE), lineRule='auto'))
        if not self.cover_mode:
            pp.append(node('snapToGrid', val=0))
        pp.append(node('keepNext', val=1 if (keep or role in ['heading', 'sub']) else 0))
        pp.append(node('jc', val='center' if center else 'left'))
        r = node('r'); pr = node('rPr')
        pr.append(node('rFonts', ascii='돋움', eastAsia='돋움', hAnsi='돋움'))
        if bold if bold is not None else role in ['heading', 'sub']:
            pr.append(node('b'))
        pr.append(node('color', val='000000'))
        pr.append(node('sz', val=size or {'heading': 24, 'caption': 20, 'ref': 18}.get(role, 22)))
        r.append(pr); t = node('t'); t.text = text
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        r.append(t); p.append(r)
        return p

    def add(self, text, role='body', **kw):
        self.body.append(self.paragraph(text, role, **kw))
        self.texts.append(text)

    def table(self, headers, records, widths, caption=None, note=None, size=18):
        self.table_count += 1
        if caption:
            self.add(f'표 {self.table_count}  {caption}', 'caption', keep=True, bold=True)
        t = node('tbl'); pr = node('tblPr')
        pr.append(node('tblW', w=sum(widths), type='dxa'))
        pr.append(node('jc', val='center'))
        borders = node('tblBorders')
        for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
            borders.append(node(side, val='single', sz=4, color='BFBFBF'))
        pr.append(borders); pr.append(node('tblLayout', type='fixed')); margins = node('tblCellMar')
        for side, wv in [('top', 30), ('left', 70), ('bottom', 30), ('right', 70)]:
            margins.append(node(side, w=wv, type='dxa'))
        pr.append(margins); t.append(pr); grid = node('tblGrid')
        for width in widths:
            grid.append(node('gridCol', w=width))
        t.append(grid)
        for index, values in enumerate([headers] + list(records)):
            tr = node('tr'); trp = node('trPr'); trp.append(node('cantSplit'))
            if index == 0:
                trp.append(node('tblHeader'))
            tr.append(trp)
            for col, (value, width) in enumerate(zip(values, widths)):
                tc = node('tc'); tcp = node('tcPr')
                tcp.append(node('tcW', w=width, type='dxa'))
                if index == 0:
                    tcp.append(node('shd', val='clear', color='auto', fill='EFEFEF'))
                tcp.append(node('vAlign', val='center'))
                tc.append(tcp)
                lines = str(value).split('\n')
                for li, line in enumerate(lines):
                    p = self.paragraph(line, size=size, keep=True)
                    pp = p.find('w:pPr', NS); sp = pp.find('w:spacing', NS)
                    sp.set('{' + W + '}after', '0'); sp.set('{' + W + '}line', '240')
                    if col > 0 or index == 0:
                        pp.find('w:jc', NS).set('{' + W + '}val', 'center')
                    if index == 0:
                        p.find('w:r/w:rPr', NS).insert(1, node('b'))
                    tc.append(p)
                tr.append(tc)
            t.append(tr)
        self.body.append(t)
        self.texts.extend([' | '.join(map(str, r)) for r in [headers] + list(records)])
        if note:
            self.add(note, 'caption')
        else:
            self.add('')

    def figure(self, path, caption, width=5.6):
        path = ROOT / path
        assert path.exists(), path
        self.image_count += 1
        with tempfile.TemporaryDirectory() as temp:
            opaque = Path(temp) / 'figure.png'
            im = Image.open(path)
            if im.mode == 'RGBA':
                assert im.getchannel('A').getextrema() == (255, 255)
            im.convert('RGB').save(opaque, dpi=(300, 300))
            doc = Document(); doc.add_paragraph().add_run().add_picture(str(opaque), width=Inches(width))
            fragment = Path(temp) / 'fragment.docx'; doc.save(fragment)
            with ZipFile(fragment) as z:
                x = E.fromstring(z.read('word/document.xml'))
                p = x.find('w:body/w:p', NS)
                image_name = next(n for n in z.namelist() if n.startswith('word/media/'))
                media = f'word/media/final36_{self.image_count}.png'
                self.parts[media] = z.read(image_name)
        used = [int(x.get('Id')[3:]) for x in self.rels if x.get('Id', '').startswith('rId') and x.get('Id', '')[3:].isdigit()]
        rid = f'rId{max(used, default=0) + 1}'
        p.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip').set('{' + R + '}embed', rid)
        props = p.find('.//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr')
        props.set('id', str(200 + self.image_count)); props.set('descr', caption)
        E.SubElement(self.rels, '{' + REL + '}Relationship', Id=rid, Type=R + '/image', Target=media.removeprefix('word/'))
        pp = node('pPr'); pp.append(node('keepNext', val=1)); pp.append(node('snapToGrid', val=0))
        pp.append(node('spacing', before=40, after=20)); pp.append(node('jc', val='center')); p.insert(0, pp)
        self.body.append(p)
        self.add(f'그림 {self.image_count}  {caption}', 'caption', center=True)
        self.images.append({'path': str(path.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha(path), 'caption': caption})

    def cover(self, summary):
        self.cover_mode = True
        for index, text in [(2, ''), (3, '김형수없는김형수'), (4, '없음'),
                            (5, 'PyTorch, segmentation_models_pytorch, timm, NumPy, OpenCV, Pillow, scikit-image, Matplotlib')]:
            cell = self.rows[index].findall('w:tc', NS)[-1]; self.clear(cell); cell.append(self.paragraph(text))
        cell = self.rows[7].find('w:tc', NS); self.clear(cell)
        for text in summary:
            cell.append(self.paragraph(text))
        self.cover_mode = False

    def flow_rows(self):
        """Replace the single body row with consecutive rows so Word paginates quickly
        and content flows without forced page breaks or trailing white space."""
        outer = self.rows[9].getparent()
        position = outer.index(self.rows[9])
        blocks = [x for x in self.body if x.tag != '{' + W + '}tcPr']
        groups, pending = [], []
        for block in blocks:
            pending.append(block)
            keep = block.tag == '{' + W + '}tbl'
            kn = block.find('w:pPr/w:keepNext', NS)
            if kn is not None and kn.get('{' + W + '}val') == '1':
                keep = True
            if not keep:
                groups.append(pending); pending = []
        if pending:
            groups.append(pending)
        for i, group in enumerate(groups):
            row = deepcopy(self.rows[9]); cell = row.find('w:tc', NS)
            row.find('w:trPr', NS).append(node('cantSplit', val=0))
            self.clear(cell)
            borders = cell.find('w:tcPr/w:tcBorders', NS)
            for side in ['top', 'bottom']:
                edge = borders.find('w:' + side, NS)
                if (side == 'top' and i > 0) or (side == 'bottom' and i < len(groups) - 1):
                    edge.set('{' + W + '}val', 'nil')
            for block in group:
                cell.append(deepcopy(block))
            if cell[-1].tag != '{' + W + '}p':
                cell.append(self.paragraph(''))
            outer.insert(position + i, row)
        outer.remove(self.rows[9])

    def flow_top_level(self):
        """Move the body out of the form cell so numeric tables and figures can break across
        pages normally. Word never splits a nested table across pages, which left large gaps."""
        table = self.rows[9].getparent()
        body = table.getparent()
        blocks = [x for x in self.body if x.tag != '{' + W + '}tcPr']
        table.remove(self.rows[9])
        position = body.index(table) + 1
        for i, block in enumerate(blocks):
            body.insert(position + i, deepcopy(block))

    def save(self):
        self.flow_top_level()
        for element in self.xml.iter():
            if element.tag == '{' + W + '}sectPr':
                continue
            for attr in list(element.attrib):
                if E.QName(attr).localname in ['paraId', 'textId'] or E.QName(attr).localname.startswith('rsid'):
                    del element.attrib[attr]
        assert E.tostring(self.xml.find('.//w:sectPr', NS)) == self.section
        self.parts['word/document.xml'] = E.tostring(self.xml, xml_declaration=True, encoding='UTF-8', standalone=True)
        self.parts['word/_rels/document.xml.rels'] = E.tostring(self.rels, xml_declaration=True, encoding='UTF-8', standalone=True)
        settings = E.fromstring(self.parts['word/settings.xml'])
        for compatibility in settings.findall('.//w:compatSetting', NS):
            if compatibility.get('{' + W + '}name') == 'compatibilityMode':
                compatibility.set('{' + W + '}val', '15')
        self.parts['word/settings.xml'] = E.tostring(settings, xml_declaration=True, encoding='UTF-8', standalone=True)
        ct = E.fromstring(self.parts['[Content_Types].xml'])
        if not any(x.get('Extension') == 'png' for x in ct):
            E.SubElement(ct, '{http://schemas.openxmlformats.org/package/2006/content-types}Default', Extension='png', ContentType='image/png')
            self.parts['[Content_Types].xml'] = E.tostring(ct, xml_declaration=True, encoding='UTF-8', standalone=True)
        with ZipFile(OUT, 'w') as z:
            for info in self.infos:
                z.writestr(info, self.parts.pop(info.filename))
            for name, value in self.parts.items():
                z.writestr(name, value)
        with ZipFile(OUT) as z, ZipFile(REFERENCE) as ref:
            fixed = [n for n in ref.namelist() if n not in ['word/document.xml', 'word/_rels/document.xml.rels', '[Content_Types].xml', 'word/settings.xml']]
            assert all(z.read(n) == ref.read(n) for n in fixed)
        WORK.mkdir(exist_ok=True, parents=True)
        (WORK / 'report_text.txt').write_text('\n\n'.join(self.texts), encoding='utf-8')
        evidence = {'output': str(OUT), 'sha256': sha(OUT), 'reference_sha256': sha(REFERENCE),
                    'preserve_only_parts_match': True, 'preserve_only_parts': fixed, 'section_unchanged': True,
                    'official_validation20_mIoU': 80.62075310543737, 'train70_internal_oof_mIoU': 80.77829756479382,
                    'images': self.images, 'tables': self.table_count, 'visual_qa_complete': False}
        (WORK / 'build_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'output': str(OUT), 'sha256': sha(OUT), 'figures': self.image_count, 'tables': self.table_count}, ensure_ascii=False))


def main():
    content = json.loads((WORK / 'content.json').read_text(encoding='utf-8'))
    f = Form(); f.cover(content['cover_summary'])
    for block in content['blocks']:
        kind = block['kind']
        if kind == 'text':
            f.add(block['text'], block.get('role', 'body'))
        elif kind == 'table':
            f.table(block['headers'], block['rows'], block['widths'], caption=block.get('caption'), note=block.get('note'), size=block.get('size', 18))
        elif kind == 'figure':
            f.figure(block['path'], block['caption'], width=block.get('width', 5.6))
        else:
            raise ValueError(block)
    f.save()


if __name__ == '__main__':
    main()
