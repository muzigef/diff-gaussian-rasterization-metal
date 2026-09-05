"""Fetch selected members of the official ZIP using HTTP Range, with ZIP CRC checks.

No extraction of archive paths: callers explicitly choose the output filename.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
import zlib

BASE = 'https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/'


def request_range(url, start, end):
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
    response = urllib.request.urlopen(request, timeout=60)
    if response.status != 206:
        response.close()
        raise RuntimeError('Server did not honor HTTP Range; refusing full archive download')
    return response


def read_range(url, start, end):
    with request_range(url, start, end) as response:
        data = response.read(end-start+2)
    if len(data) != end-start+1:
        raise RuntimeError('Incomplete range response')
    return data


def compressed_parts(url, start, length, destination):
    """Cache independent ranges so a slow/interrupted connection need not restart the model."""
    chunk_size = 4 * 1024 * 1024
    cache = destination.parent / '.downloads'
    cache.mkdir(parents=True, exist_ok=True)
    ranges = [(offset, min(chunk_size, length-offset)) for offset in range(0, length, chunk_size)]

    def download(item):
        offset, count = item
        key = hashlib.sha256(f'{url}:{start+offset}:{count}'.encode()).hexdigest()
        path = cache / key
        if not path.exists() or path.stat().st_size != count:
            temporary = path.with_suffix('.partial')
            for attempt in range(3):
                try:
                    data = read_range(url, start+offset, start+offset+count-1)
                    temporary.write_bytes(data)
                    temporary.replace(path)
                    break
                except (OSError, RuntimeError):
                    if attempt == 2:
                        raise
            print(f'Cached range {offset+count:,}/{length:,}', flush=True)
        return path

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(download, ranges))
    for path in paths:
        yield path.read_bytes()


def directory(url):
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
        size = int(response.headers['Content-Length'])
    tail = read_range(url, max(0, size-65536), size-1)
    pos = tail.rfind(b'PK\x05\x06')
    if pos < 0: raise RuntimeError('ZIP end record missing')
    fields = struct.unpack_from('<4s4H2IH', tail, pos)
    length, offset = fields[5:7]
    if offset == 0xffffffff:
        pos64 = tail.rfind(b'PK\x06\x06')
        if pos64 < 0: raise RuntimeError('ZIP64 end record missing')
        fields64 = struct.unpack_from('<4sQ2H2I4Q', tail, pos64)
        length, offset = fields64[-2:]
    central = read_range(url, offset, offset+length-1)
    entries = {}
    pos = 0
    while central[pos:pos+4] == b'PK\x01\x02':
        v = struct.unpack_from('<4s6H3I5H2I', central, pos)
        nl, xl, cl = v[10:13]
        name = central[pos+46:pos+46+nl].decode('utf-8')
        compressed, expanded, offset = v[8], v[9], v[16]
        extra = central[pos+46+nl:pos+46+nl+xl]
        e = 0
        while e+4 <= len(extra):
            tag, count = struct.unpack_from('<HH', extra, e)
            if tag == 1:
                cursor = e+4
                values = [expanded, compressed, offset]
                for i in range(3):
                    if values[i] == 0xffffffff:
                        values[i] = struct.unpack_from('<Q', extra, cursor)[0]; cursor += 8
                expanded, compressed, offset = values
            e += count+4
        entries[name] = dict(compressed=compressed, expanded=expanded, offset=offset, crc=v[7], method=v[4])
        pos += 46+nl+xl+cl
    return entries


def extract(url, name, destination, entries):
    destination = Path(destination)
    entry = entries[name]
    header = read_range(url, entry['offset'], entry['offset']+29)
    if header[:4] != b'PK\x03\x04': raise RuntimeError('Invalid local ZIP header')
    nl, xl = struct.unpack_from('<HH', header, 26)
    start = entry['offset']+30+nl+xl
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.partial')
    decoder = zlib.decompressobj(-15) if entry['method'] == 8 else None
    if entry['method'] not in (0,8): raise RuntimeError('Unsupported ZIP compression')
    crc = 0; written = 0; digest = hashlib.sha256()
    print(f'Downloading {name}: {entry["compressed"]:,} bytes', flush=True)
    with temporary.open('wb') as output:
        for chunk in compressed_parts(url, start, entry['compressed'], destination):
            chunk = decoder.decompress(chunk) if decoder else chunk
            output.write(chunk); crc = zlib.crc32(chunk, crc); digest.update(chunk); written += len(chunk)
        if decoder:
            chunk = decoder.flush(); output.write(chunk); crc = zlib.crc32(chunk,crc); digest.update(chunk); written += len(chunk)
    if written != entry['expanded'] or crc != entry['crc']:
        temporary.unlink(missing_ok=True)
        raise RuntimeError('ZIP size/CRC verification failed')
    temporary.replace(destination)
    print(f'Verified {destination}', flush=True)
    return dict(url=url, member=name, bytes=written, sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('output/public/train'))
    parser.add_argument('--iteration', choices=('7000','30000'), default='7000')
    parser.add_argument('--images', action='store_true', help='Also fetch the first official test render and GT')
    args = parser.parse_args()
    url = BASE+'datasets/pretrained/models.zip'
    entries = directory(url)
    manifest = []
    for member, filename in [('train/cameras.json','cameras.json'),('train/cfg_args','cfg_args'),
                             (f'train/point_cloud/iteration_{args.iteration}/point_cloud.ply','point_cloud.ply')]:
        manifest.append(extract(url,member,args.output/filename,entries))
    (args.output/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.images:
        url = BASE+'evaluation/images.zip'
        entries = directory(url)
        camera_name = json.loads((args.output/'cameras.json').read_text())[0]['img_name']
        # The 2023 public archive uses source image names (00001, 00009, ...).
        # Current render.py uses enumeration indices; do not infer this archive's names from that code.
        manifest = [extract(url, f'train/test/ours_{args.iteration}/{kind}/{camera_name}.png',
                            args.output/(kind+'.png'), entries) for kind in ('renders', 'gt')]
        for record in manifest:
            record['camera_name'] = camera_name
        (args.output/'image_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__ == '__main__': main()
