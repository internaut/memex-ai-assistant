import logging
import random
import sys
import multiprocessing as mp
import string
import pickle
import hashlib
import tempfile
from functools import partial
from getpass import getpass
from io import BytesIO
from pathlib import Path
from argparse import ArgumentParser
from tarfile import TarFile
from logging import getLogger

import pgpy
from sentence_transformers import CrossEncoder
import transformers
import torch


HERE = Path(__file__).parent.resolve()
DEFAULT_ENCODING = 'utf-8'
DEFAULT_MEMEXDB_PATH = HERE / 'memexdb'
DEFAULT_GPG_PRIV_KEY = Path().home() / '.gnupg' / 'default-sec.asc'
DEFAULT_EMBEDDINGS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_NUM_CONTEXT_DOCS = 20
DEFAULT_N_WORKERS = mp.cpu_count() - 1
LLM_MODEL_PATH = (HERE / '..' / 'llm_models' / 'Llama-3.2-3B-Instruct').resolve()

CONTEXT_DOC_TEMPLATE = """$date
----------

$text

///
"""

INSTRUCTIONS_TEMPLATE = """Die folgenden Texte sind Tagebucheinträge. Sie sind nicht sortiert und starten mit einem 
Datum im Jahr-Monat-Tag Format, gefolgt von "----------" und im Anschluss dem Texteintrag. Jeder Tagebucheintrag endet
mit "///". Beantworte alle Fragen in Bezug auf die Tagebucheinträge. Hier sind die Tagebucheinträge:

$documents
"""


def printerr(*args):
    """Shortcut to print to stdout."""
    print(*args, file=sys.stderr)


def load_documents(memexdb, num_workers, encoding, key, key_passwd, sample, use_cache):
    memexdb_path = Path(memexdb)
    if not memexdb_path.exists():
        printerr(f'memexdb path does not exist: "{memexdb_path}"')
        exit(1)

    memexfiles = sorted(memexdb_path.glob('*/*.tar.gpg'))
    logger.info(f'found {len(memexfiles)} documents')

    if use_cache:
        h = hashlib.new('sha256')
        h.update(''.join(map(str, memexfiles)).encode() + str(sample).encode())
        cachefile = Path(tempfile.gettempdir()) / f'{h.hexdigest()}.pickle'
    else:
        cachefile = None

    if cachefile and cachefile.exists():
        logger.info(f'loading documents from cache')
        with open(cachefile, 'rb') as f:
            doc_dates, docs = pickle.load(f)
    else:
        if sample:
            sample = min(sample, len(memexfiles))
            logger.info(f'will sample {sample} documents')
            memexfiles = random.sample(memexfiles, sample)

        logger.info(f'loading and decrypting {len(memexfiles)} documents from memexdb at "{memexdb_path}" using '
                    f'{num_workers} workers')

        decrypt_memex_w_args = partial(decrypt_doc, encoding=encoding, key_fpath=key, key_passwd=key_passwd)

        with mp.Pool(num_workers) as proc_pool:
            res = proc_pool.map(decrypt_memex_w_args, memexfiles)
            doc_dates, docs = zip(*res)

        if cachefile:
            logger.info(f'saving documents to cache')
            with open(cachefile, 'wb') as f:
                pickle.dump((doc_dates, docs), f)

    return doc_dates, docs


def decrypt_doc(fpath, encoding, key_fpath, key_passwd):
    date = fpath.name.removesuffix('.tar.gpg')

    msg = pgpy.PGPMessage.from_file(fpath)
    key_loaded, _ = pgpy.PGPKey.from_file(key_fpath)

    if key_loaded.is_unlocked:
        decrypted = key_loaded.decrypt(msg).message
    else:
        with key_loaded.unlock(key_passwd) as key:
            decrypted = key.decrypt(msg).message

    tar = TarFile(fileobj=BytesIO(decrypted))
    try:
        txtbuffer = tar.extractfile(f'./{date}.txt')
    except KeyError:
        raise SystemError(f'could not find memex text file in tar archive inside "{fpath}"')
    return date, txtbuffer.read().decode(encoding)


def process_input(initial_query):
    if initial_query:
        q = initial_query
    else:
        q = input('Please enter your query. Enter "quit" to stop the program.\n\n')

    q = q.strip()
    logger.info(f'user query: {q}')

    if q.lower() == 'quit':
        return False

    print(f'Finding the {args.num_context_docs} most relevant documents out of {len(docs)} documents ...')
    logger.info(f'ranking {len(docs)} documents using {args.num_workers} workers')
    ranks = model.rank(q, docs, top_k=args.num_context_docs, num_workers=args.num_workers, show_progress_bar=True)

    context_documents = []
    context_templ = string.Template(CONTEXT_DOC_TEMPLATE)
    for rank in ranks:
        index = rank['corpus_id']
        context_documents.append(context_templ.substitute(date=doc_dates[index], text=docs[index]))

    instructions_templ = string.Template(INSTRUCTIONS_TEMPLATE)
    instructions = instructions_templ.substitute(documents='\n'.join(context_documents))
    print(instructions)

    return True


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('query', nargs='?', default=None)
    argparser.add_argument('--key', default=str(DEFAULT_GPG_PRIV_KEY))
    argparser.add_argument('--keypw', default='')
    argparser.add_argument('--memexdb', default=str(DEFAULT_MEMEXDB_PATH))
    argparser.add_argument('--num-context-docs', type=int, default=DEFAULT_NUM_CONTEXT_DOCS)
    argparser.add_argument('--num-workers', type=int, default=DEFAULT_N_WORKERS)
    argparser.add_argument('--sample', type=int, default=0)
    argparser.add_argument('--encoding', default=DEFAULT_ENCODING)
    argparser.add_argument('--verbose', action='store_true')
    argparser.add_argument('--use-cache', action='store_true')

    args = argparser.parse_args()

    print(f'Starting up the assistent ...')

    logging.basicConfig()
    logger = getLogger(__name__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f'using GPG key "{args.key}"')
    key, _ = pgpy.PGPKey.from_file(args.key)
    if args.keypw:
        key_passwd = args.keypw
    else:
        key_passwd = ''

    if not key.is_unlocked:
        if not key_passwd:
            key_passwd = getpass('The provided key is locked with a password. Please provide the password: ')

        try:
            with key.unlock(key_passwd) as _:
                pass
        except pgpy.errors.PGPDecryptionError:
            print('The provided key password is incorrect.')
            exit(1)

    logger.info(f'loading embeddings model "{DEFAULT_EMBEDDINGS_MODEL}"')
    model = CrossEncoder(DEFAULT_EMBEDDINGS_MODEL)

    print('Loading memex documents ...')
    doc_dates, docs = load_documents(
        memexdb=args.memexdb,
        num_workers=args.num_workers,
        encoding=args.encoding,
        key=args.key,
        key_passwd=key_passwd,
        sample=args.sample,
        use_cache=args.use_cache
    )

    cont = True
    initial_query = args.query
    while cont:
        cont = process_input(initial_query)
        initial_query = None

    print('Done.')
