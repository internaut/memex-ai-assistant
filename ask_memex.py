import logging
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


def decrypt_memex(fpath, encoding, key_fpath, key_passwd):
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


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('query', nargs='?', default=None)
    argparser.add_argument('--key', default=str(DEFAULT_GPG_PRIV_KEY))
    argparser.add_argument('--keypw', default='')
    argparser.add_argument('--memexdb', default=str(DEFAULT_MEMEXDB_PATH))
    argparser.add_argument('--num-context-docs', type=int, default=DEFAULT_NUM_CONTEXT_DOCS)
    argparser.add_argument('--num-workers', type=int, default=DEFAULT_N_WORKERS)
    argparser.add_argument('--encoding', default=DEFAULT_ENCODING)
    argparser.add_argument('--verbose', action='store_true')
    argparser.add_argument('--use-cache', action='store_true')

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

    if args.use_cache:
        h = hashlib.new('sha256')
        h.update(q.encode())
        q_hash = h.hexdigest()
        cachefile = Path(tempfile.gettempdir()) / f'{q_hash}.pickle'
    else:
        cachefile = None

    if cachefile and cachefile.exists():
        logger.info(f'loading documents from cache')
        with open(cachefile, 'rb') as f:
            doc_dates, docs = pickle.load(f)
    else:
        memexdb_path = Path(args.memexdb)
        logger.info(f'loading and decrypting documents from memexdb at "{memexdb_path}" using '
                    f'{args.num_workers} workers')
        if not memexdb_path.exists():
            printerr(f'memexdb path does not exist: "{memexdb_path}"')
            exit(1)

        memexfiles = sorted(memexdb_path.glob('*/*.tar.gpg'))
        logger.info(f'found {len(memexfiles)} documents')
        decrypt_memex_w_args = partial(decrypt_memex, encoding=args.encoding, key_fpath=args.key, key_passwd=key_passwd)

        with mp.Pool(args.num_workers) as proc_pool:
            res = proc_pool.map(decrypt_memex_w_args, memexfiles, len(memexfiles) // args.num_workers + 1)
            doc_dates, docs = zip(*res)

        if cachefile:
            logger.info(f'saving documents to cache')
            with open(cachefile, 'wb') as f:
                pickle.dump((doc_dates, docs), f)

    logger.info(f'ranking documents')
    ranks = model.rank(q, docs, top_k=args.num_context_docs, num_workers=args.num_workers)

    context_documents = []
    context_templ = string.Template(CONTEXT_DOC_TEMPLATE)
    for rank in ranks:
        index = rank['corpus_id']
        context_documents.append(context_templ.substitute(date=doc_dates[index], text=docs[index]))

    instructions_templ = string.Template(INSTRUCTIONS_TEMPLATE)
    instructions = instructions_templ.substitute(documents='\n'.join(context_documents))
    print(instructions)

    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=str(LLM_MODEL_PATH),
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     device_map="auto",
    # )
    #
    # messages = [
    #     {"role": "system", "content": instructions},
    #     {"role": "user", "content": q},
    # ]
    #
    # outputs = pipeline(
    #     messages,
    #     max_new_tokens=256,
    # )
    # print(outputs[0]["generated_text"][-1])

    logger.info('done')
