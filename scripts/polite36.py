"""Convert plain-style (-다) sentence endings in the report content to the polite (-습니다) style.

Only the final 다 of a sentence is touched (followed by an optional citation or parenthesis and a period,
or the end of the string). Particles such as 보다 inside sentences are never changed.
Usage: python scripts/polite36.py [--apply]   (without --apply it only prints before/after endings)
"""
import json
import re
import sys
from pathlib import Path

SRC = Path('reports/final36_document_work/content.json')
BACKUP = Path('reports/final36_document_work/content_plain_backup.json')
SS, N, B = 20, 4, 17                      # 받침: ㅆ, ㄴ, ㅂ
VOWEL_STEMS = set('하크르되')              # 하다 크다 다르다 되다 -> 합니다 큽니다 다릅니다 됩니다
# 명사 + 이다가 줄어든 형태 중 '이'로 끝나는 명사 (차이다 -> 차이입니다)
NOUN_ENDING_I = {'차이', '높이', '간격차이'}


def hangul(c):
    return '가' <= c <= '힣'


def split(c):
    o = ord(c) - 0xAC00
    return o // 588, (o % 588) // 28, o % 28


def join(a, b, c):
    return chr(0xAC00 + a * 588 + b * 28 + c)


def polite(w):
    last = w[-1]
    if w.endswith('아니'):
        return w[:-1] + '닙니다'
    if not hangul(last):                  # 1.6%다, 비교)다 : 숫자·기호 뒤의 이다
        return w + '입니다'
    a, b, c = split(last)
    if c == SS:                           # 했다 았다 었다 줄었다
        return w + '습니다'
    if last == '는':                      # 않는다 돕는다 남는다
        return w[:-1] + '습니다'
    if c == N:                            # 한다 친다 된다 오른다
        return w[:-1] + join(a, b, B) + '니다'
    if c == 0:
        if last in VOWEL_STEMS:
            return w[:-1] + join(a, b, B) + '니다'
        if last == '이':
            return w + '입니다' if w in NOUN_ENDING_I else w[:-1] + '입니다'
        return w + '입니다'               # 목표다 구조다 가지다 (이다의 축약)
    return w + '습니다'                   # 있다 같다 넓다 많다


PAT = re.compile(r'(\S+)다((?:\[[\d,]+\]|\([^)\n]*\))?)(?=\.(?:\s|$)|$)')


def convert(text):
    return PAT.sub(lambda m: polite(m.group(1)) + m.group(2), text)


def walk(obj, fn, path=''):
    """text 블록(heading/sub/ref 제외), 표 셀, 캡션, cover_summary의 문자열에 fn을 적용한다."""
    if isinstance(obj, dict):
        kind, role = obj.get('kind'), obj.get('role')
        for k, v in list(obj.items()):
            if k == 'text' and kind == 'text' and role not in ('heading', 'sub', 'ref'):
                obj[k] = fn(v)
            elif k in ('caption', 'note') and isinstance(v, str):
                obj[k] = fn(v)
            elif k == 'rows':
                obj[k] = [[fn(cell) for cell in row] for row in v]
            elif isinstance(v, (dict, list)):
                walk(v, fn)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                walk(v, fn)


def main():
    apply = '--apply' in sys.argv
    sys.stdout.reconfigure(encoding='utf-8')
    data = json.loads((BACKUP if BACKUP.exists() else SRC).read_text(encoding='utf-8'))
    changes = []

    def fn(t):
        out = convert(t)
        for m in PAT.finditer(t):
            changes.append((t[max(0, m.start(1) - 6):m.end(1) + 1 + len(m.group(2))], polite(m.group(1)) + m.group(2)))
        return out
    data['cover_summary'] = [fn(t) for t in data['cover_summary']]
    walk(data['blocks'], fn)
    seen = set()
    for old, new in changes:
        if (old, new) not in seen:
            seen.add((old, new)); print(f'{old!r:40} -> {new!r}')
    print('total changes:', len(changes))
    if apply:
        SRC.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        print('applied to', SRC)


if __name__ == '__main__':
    main()
