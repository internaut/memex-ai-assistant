import sys
from io import BytesIO
from pathlib import Path
from argparse import ArgumentParser
from tarfile import TarFile

from gnupg import GPG


DEFAULT_ENCODING = 'utf-8'
DEFAULT_MEMEXDB_PATH = Path(__file__).parent.resolve() / 'memexdb'


def printerr(*args):
    """Shortcut to print to stdout."""
    print(*args, file=sys.stderr)


def decrypt_memex(fpath, encoding):
    date = fpath.name.removesuffix('.tar.gpg')
    with open(fpath, 'rb') as f:
        decrypted = gpg.decrypt_file(f)
        if decrypted.ok:
            tar = TarFile(fileobj=BytesIO(decrypted.data))
            try:
                txtbuffer = tar.extractfile(f'./{date}.txt')
            except KeyError:
                raise SystemError(f'could not find memex text file in tar archive inside "{fpath}"')
            return txtbuffer.read().decode(encoding)
        else:
            raise SystemError(f'could not decrypt file "{fpath}"; details: {decrypted.status}')


argparser = ArgumentParser()
argparser.add_argument('query')
argparser.add_argument('--memexdb', default=str(DEFAULT_MEMEXDB_PATH))
argparser.add_argument('--encoding', default=DEFAULT_ENCODING)

args = argparser.parse_args()

gpg = GPG()

memexdb_path = Path(args.memexdb)

if not memexdb_path.exists():
    printerr(f'memexdb path does not exist: "{memexdb_path}"')
    exit(1)

for fpath in sorted(memexdb_path.glob('*/*.tar.gpg')):
    print(fpath)
    doc = decrypt_memex(fpath, encoding=args.encoding)
    print(doc)
    break