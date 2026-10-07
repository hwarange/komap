"""MICAST LOM 학습 데이터에서 필요한 파일만 HTTP Range로 병렬 다운로드 (zip 안의 zip을 통째로 받지 않음)."""
import io, os, sys, struct, time, zipfile, zlib, urllib.request
from concurrent.futures import ThreadPoolExecutor
URL0 = 'https://zenodo.org/records/18740450/files/Model_training.zip?download=1'
OUT = os.path.expanduser('~/komap/external/micast')

def get(url, a, b, tries=12):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b}'})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            time.sleep(min(60, 3 * 2 ** min(t, 5)))   # 요청 제한(429)이면 점점 길게 기다림
    raise RuntimeError(f'range {a}-{b} failed')

class RangeFile(io.RawIOBase):
    """원격 파일의 [base, base+size) 구간을 하나의 파일처럼 읽는다."""
    def __init__(self, url, base, size): self.url, self.base, self.size, self.pos = url, base, size, 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos
    def readinto(self, b):
        if self.pos >= self.size: return 0
        n = min(len(b), self.size - self.pos)
        d = get(self.url, self.base + self.pos, self.base + self.pos + n - 1)
        b[:len(d)] = d; self.pos += len(d); return len(d)

with urllib.request.urlopen(urllib.request.Request(URL0, method='HEAD')) as r:
    URL, TOTAL = r.geturl(), int(r.headers['Content-Length'])
outer = zipfile.ZipFile(io.BufferedReader(RangeFile(URL, 0, TOTAL), buffer_size=1 << 16))
zi = outer.getinfo('Model_training/training_dataset_Al-Fe-Si_LOM.zip')
lh = get(URL, zi.header_offset, zi.header_offset + 29)
nlen, elen = struct.unpack('<HH', lh[26:30])
start = zi.header_offset + 30 + nlen + elen
print('inner LOM zip: start', start, 'size %.2f GB' % (zi.compress_size / 1e9), flush=True)
inner = zipfile.ZipFile(io.BufferedReader(RangeFile(URL, start, zi.compress_size), buffer_size=1 << 16))
infos = inner.infolist()
print('inner files', len(infos), flush=True)
import collections
c = collections.Counter((os.path.dirname(i.filename), i.filename.rsplit('.', 1)[-1]) for i in infos if not i.is_dir())
for k, v in sorted(c.items()): print(v, k)
for i in infos:
    if i.filename.endswith('.json'): print('json', i.filename, round(i.file_size / 1e6, 1), 'MB', 'compress', i.compress_type)
png = [i for i in infos if i.filename.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))]
print('image sample', png[0].filename, round(png[0].file_size / 1e6, 2), 'MB', 'compress', png[0].compress_type, flush=True)

def fetch(i):
    """멤버 하나: 로컬 헤더 + 데이터를 range로 받아 저장."""
    dst = os.path.join(OUT, 'lom', i.filename)
    if os.path.exists(dst) and os.path.getsize(dst) == i.file_size: return i.file_size
    h = get(URL, start + i.header_offset, start + i.header_offset + 29)
    n, e = struct.unpack('<HH', h[26:30])
    a = start + i.header_offset + 30 + n + e
    d = get(URL, a, a + i.compress_size - 1)
    if i.compress_type == zipfile.ZIP_DEFLATED: d = zlib.decompress(d, -15)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, 'wb').write(d); return len(d)

mode = sys.argv[1] if len(sys.argv) > 1 else "peek"
if mode == 'peek':
    todo = [i for i in infos if i.filename.endswith('.json')] + png[:4]
elif mode == 'train':   # 학습 사진 앞에서 N장만 (json은 이미 받음)
    N = int(sys.argv[2])
    todo = sorted([i for i in png if '/TRAIN/' in '/' + i.filename], key=lambda i: int(os.path.basename(i.filename)[5:-4]))[:N]
else:   # full: json 전부 + 이미지 N장 (학습 앞쪽부터) + 검증 이미지 전부
    N = int(sys.argv[2])
    tr = [i for i in png if '/TRAIN/' in '/' + i.filename or 'train' in os.path.basename(i.filename).lower()][:N]
    te = [i for i in png if '/TEST/' in '/' + i.filename or 'test' in os.path.basename(i.filename).lower()]
    todo = [i for i in infos if i.filename.endswith('.json')] + tr + te
t0 = time.time(); tot = 0
with ThreadPoolExecutor(8) as ex:
    for k, s in enumerate(ex.map(fetch, todo)):
        tot += s
        if k % 200 == 0: print(f'{k}/{len(todo)} files, {tot/1e6:.0f} MB, {tot/1e6/max(time.time()-t0,1e-9):.1f} MB/s', flush=True)
print(f'done {len(todo)} files, {tot/1e9:.2f} GB in {time.time()-t0:.0f}s ({tot/1e6/(time.time()-t0):.1f} MB/s)', flush=True)
