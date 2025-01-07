import logging
import os
import sys
import multiprocessing as mp
from functools import partial
from io import BytesIO
from pathlib import Path
from argparse import ArgumentParser
from tarfile import TarFile
from logging import getLogger

from gnupg import GPG
from sentence_transformers import CrossEncoder


DEFAULT_ENCODING = 'utf-8'
DEFAULT_MEMEXDB_PATH = Path(__file__).parent.resolve() / 'memexdb'
DEFAULT_EMBEDDINGS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_NUM_CONTEXT_DOCS = 10
DEFAULT_N_WORKERS = mp.cpu_count() - 1


def printerr(*args):
    """Shortcut to print to stdout."""
    print(*args, file=sys.stderr)


def decrypt_memex(fpath, encoding):
    gpg = GPG()
    date = fpath.name.removesuffix('.tar.gpg')
    print(f'process {os.getpid()} is memex for date {date}')

    with open(fpath, 'rb') as f:
        decrypted = gpg.decrypt_file(f)
        if decrypted.ok:
            tar = TarFile(fileobj=BytesIO(decrypted.data))
            try:
                txtbuffer = tar.extractfile(f'./{date}.txt')
            except KeyError:
                raise SystemError(f'could not find memex text file in tar archive inside "{fpath}"')
            return date, txtbuffer.read().decode(encoding)
        else:
            raise SystemError(f'could not decrypt file "{fpath}"; details: {decrypted.status}')


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('query', nargs='?', default=None)
    argparser.add_argument('--memexdb', default=str(DEFAULT_MEMEXDB_PATH))
    argparser.add_argument('--num-context-docs', type=int, default=DEFAULT_NUM_CONTEXT_DOCS)
    argparser.add_argument('--num-workers', type=int, default=DEFAULT_N_WORKERS)
    argparser.add_argument('--encoding', default=DEFAULT_ENCODING)
    argparser.add_argument('--verbose', action='store_true')

    args = argparser.parse_args()

    logging.basicConfig()
    logger = getLogger(__name__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.query:
        q = args.query
    else:
        q = input('Please enter your query: ')

    q = q.strip()

    logger.info(f'user query: {q}')

    logger.info(f'loading embeddings model "{DEFAULT_EMBEDDINGS_MODEL}"')
    model = CrossEncoder(DEFAULT_EMBEDDINGS_MODEL)

    memexdb_path = Path(args.memexdb)
    logger.info(f'loading and decrypting documents from memexdb at "{memexdb_path}" using {args.num_workers} workers')
    if not memexdb_path.exists():
        printerr(f'memexdb path does not exist: "{memexdb_path}"')
        exit(1)

    memexfiles = sorted(memexdb_path.glob('*/*.tar.gpg'))[:40]
    logger.info(f'found {len(memexfiles)} documents')
    decrypt_memex_w_enc = partial(decrypt_memex, encoding=args.encoding)

    with mp.Pool(args.num_workers) as proc_pool:
        res = proc_pool.map(decrypt_memex_w_enc, memexfiles, len(memexfiles) // args.num_workers + 1)
        doc_dates, docs = zip(*res)

    logger.info(f'ranking documents')
    ranks = model.rank(q, docs, top_k=args.num_context_docs, num_workers=args.num_workers)

    for rank in ranks:
        index = rank['corpus_id']
        print(f"{rank['score']:.2f}\t{doc_dates[index]}\t{docs[index]}")

    logger.info('done')
