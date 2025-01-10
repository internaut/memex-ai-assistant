import logging
import os
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
from threading import Thread

import pgpy
from sentence_transformers import CrossEncoder
import transformers
import torch


HERE = Path(__file__).parent.resolve()
DEFAULT_ENCODING = 'utf-8'
DEFAULT_MEMEXDB_PATH = HERE / 'memexdb'
DEFAULT_GPG_PRIV_KEY = Path().home() / '.gnupg' / 'default-sec.asc'
DEFAULT_EMBEDDINGS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_LANG_DETECT_MODEL = "papluca/xlm-roberta-base-language-detection"
DEFAULT_NUM_CONTEXT_DOCS = 20
DEFAULT_N_WORKERS = mp.cpu_count() - 1
LLM_MODEL_PATH = (HERE / '..' / 'llm_models' / 'Llama-3.2-1B-Instruct').resolve()
MAX_OUTPUT_TOKENS = 2048

CONTEXT_DOC_TEMPLATE = """$date
----------

$text

///
"""

INSTRUCTIONS_TEMPLATE = {
    'en': """You are a helpful assistant.
    
The following texts are diary entries. They are not sorted and start with a date in year-month-day format, 
followed by "----------" and then the text entry. Each diary entry ends with "///". Answer any questions related to the
journal entries. Here are the journal entries:

$documents
""",
    'de': """Du bist ein hilfreicher Assistent.

Die folgenden Texte sind Tagebucheinträge. Sie sind nicht sortiert und starten mit einem 
Datum im Jahr-Monat-Tag Format, gefolgt von "----------" und im Anschluss dem Texteintrag. Jeder Tagebucheintrag endet
mit "///". Beantworte alle Fragen in Bezug auf die Tagebucheinträge. Hier sind die Tagebucheinträge:

$documents
"""
}


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def printerr(*args):
    """Shortcut to print to stdout."""
    print(*args, file=sys.stderr)


def add_message(messages, role, msg):
    messages.append({"role": role, "content": msg})


def add_system_message(messages, msg):
    messages.append({"role": 'system', "content": msg})


def add_user_message(messages, msg):
    messages.append({"role": 'user', "content": msg})


def add_assistant_message(messages, msg):
    messages.append({"role": 'assistant', "content": msg})


def load_documents(memexdb, num_workers, encoding, key, key_passwd, sample, use_cache):
    memexdb_path = Path(memexdb)
    if not memexdb_path.exists():
        printerr(f'The memexdb path does not exist: "{memexdb_path}"')
        exit(1)

    memexfiles = sorted(memexdb_path.glob('*/*.tar.gpg'))
    if not memexfiles:
        printerr(f'Could not find any memex files in memexdb at "{memexdb_path}".')
        exit(1)
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


def generate_system_instructions(query, doc_dates, docs, crossenc, num_context_docs, lang, num_workers):
    print(f'Finding the {num_context_docs} most relevant documents out of {len(docs)} documents ...')
    logger.info(f'ranking {len(docs)} documents using {num_workers} workers')
    ranks = crossenc.rank(query, docs, top_k=num_context_docs, num_workers=num_workers, show_progress_bar=True)

    context_documents = []
    context_templ = string.Template(CONTEXT_DOC_TEMPLATE)
    for rank in ranks:
        index = rank['corpus_id']
        context_documents.append(context_templ.substitute(date=doc_dates[index], text=docs[index]))

    instructions_templ = string.Template(INSTRUCTIONS_TEMPLATE[lang])
    return instructions_templ.substitute(documents='\n'.join(context_documents))


def process_input(initial_query, messages, doc_dates, docs, tokenizer, crossenc, llm, num_context_docs, lang,
                  num_workers):
    if initial_query:
        q = initial_query
    else:
        q = input('Please enter your query. Enter "quit" to stop the program.\n\n')

    q = q.strip()
    logger.info(f'user query: {q}')

    if q.lower() == 'quit':
        return False, []

    instructions = generate_system_instructions(q, doc_dates, docs, crossenc, num_context_docs, lang, num_workers)
    add_system_message(messages, instructions)

    add_user_message(messages, q)

    messages_transformed = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_data = tokenizer(messages_transformed, return_tensors="pt").to(llm.device)

    streamer = transformers.TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=None)
    thread = Thread(target=llm.generate, kwargs=dict(input_data, streamer=streamer, max_new_tokens=MAX_OUTPUT_TOKENS))

    print('Assistant:')
    thread.start()
    generated_text = ""
    for new_text in streamer:
        print(new_text, end='')
        generated_text += new_text

    thread.join()
    add_assistant_message(messages, generated_text)
    print()

    return True, messages


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

    logger.info(f'loading CrossEncoder model "{DEFAULT_EMBEDDINGS_MODEL}"')
    crossenc = CrossEncoder(DEFAULT_EMBEDDINGS_MODEL)

    logger.info(f'loading language detection model "{DEFAULT_LANG_DETECT_MODEL}"')
    lang_classifier = transformers.pipeline("text-classification", model=DEFAULT_LANG_DETECT_MODEL, use_fast=False)

    logger.info(f'loading LLM and tokenizer "{LLM_MODEL_PATH}"')
    llm = transformers.AutoModelForCausalLM.from_pretrained(
        str(LLM_MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(LLM_MODEL_PATH))

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

    lang_classif_res = lang_classifier(docs[0], top_k=1, truncation=True)
    lang = 'en'
    if lang_classif_res:
        lang = lang_classif_res[0]['label']
    print(f'Detected document language "{lang}".')

    if lang not in INSTRUCTIONS_TEMPLATE:
        printerr(f'Language not supported: "{lang}".')
        exit(1)

    cont = True
    messages = []
    initial_query = args.query
    while cont:
        cont, messages = process_input(initial_query, messages, doc_dates, docs,
                                       crossenc=crossenc,
                                       tokenizer=tokenizer,
                                       llm=llm,
                                       num_context_docs=args.num_context_docs,
                                       lang=lang,
                                       num_workers=args.num_workers)
        initial_query = None

    print('Done.')
