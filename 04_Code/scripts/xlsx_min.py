"""Minimal reader for .xlsx sheets (cached cell values only) without openpyxl: the workbook is a zip of XML parts."""
import zipfile, re
import xml.etree.ElementTree as ET

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def _col(ref):
    s = re.match(r'[A-Z]+', ref).group(0); n = 0
    for ch in s:
        n = n * 26 + ord(ch) - 64
    return n - 1


def sheets(path):
    """Ordered sheet names and their part paths."""
    with zipfile.ZipFile(path) as z:
        wb = ET.fromstring(z.read('xl/workbook.xml'))
        rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        target = {r.get('Id'): r.get('Target') for r in rels}
        out = []
        for s in wb.find('m:sheets', NS):
            t = target[s.get('{%s}id' % NS['r'])].lstrip('/')
            out.append((s.get('name'), t if t.startswith('xl/') else 'xl/' + t))
        return out


def read(path, sheet):
    """Rows of one sheet as lists of strings/floats (None for empty), using cached values."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS):
                shared.append(''.join(t.text or '' for t in si.iter('{%s}t' % NS['m'])))
        part = dict(sheets(path))[sheet]
        root = ET.fromstring(z.read(part))
        rows = []
        for r in root.iter('{%s}row' % NS['m']):
            vals = {}
            for c in r.findall('m:c', NS):
                t = c.get('t'); v = c.find('m:v', NS); is_ = c.find('m:is', NS)
                if t == 's' and v is not None:
                    val = shared[int(v.text)]
                elif t == 'inlineStr' and is_ is not None:
                    val = ''.join(x.text or '' for x in is_.iter('{%s}t' % NS['m']))
                elif v is not None:
                    try:
                        val = float(v.text)
                    except ValueError:
                        val = v.text
                else:
                    continue
                vals[_col(c.get('r'))] = val
            if vals:
                row = [None] * (max(vals) + 1)
                for k, x in vals.items():
                    row[k] = x
                rows.append(row)
            else:
                rows.append([])
        return rows
