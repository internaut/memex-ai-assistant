"""
AI assistant for encrypted *memex* notes.

Uses an RAG approach with CrossEncoder to fetch the relevant documents and Llama 3.2 as LLM for text generation.

January 2025
Author: Markus Konrad <post@mkonrad.net>
"""

import logging
import random
import sys
import multiprocessing as mp
import string
import pickle
import hashlib
import tempfile
import textwrap
from time import time
from functools import partial
from getpass import getpass
from io import BytesIO
from pathlib import Path
from argparse import ArgumentParser
from tarfile import TarFile
from logging import getLogger

import pgpy
from sentence_transformers import CrossEncoder
from llama_cpp import Llama
import transformers


HERE = Path(__file__).parent.resolve()
DEFAULT_ENCODING = "utf-8"
DEFAULT_MEMEXDB_PATH = HERE / "memexdb"
DEFAULT_GPG_PRIV_KEY = Path().home() / ".gnupg" / "default-sec.asc"
DEFAULT_EMBEDDINGS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_LANG_DETECT_MODEL = "papluca/xlm-roberta-base-language-detection"
DEFAULT_NUM_CONTEXT_DOCS = 20
DEFAULT_N_WORKERS = mp.cpu_count() - 1
DEFAULT_MAX_INPUT_TOKENS = 20480
CROSS_ENC_BATCH_SIZE = 2
NO_ANSW_SENTINEL = "NO_ANSWER"
LLM_LOAD_PARAMS = dict(
    repo_id="hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF", filename="*q8_0.gguf"
)
# LLM_LOAD_PARAMS = dict(
#     repo_id="hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF",
#     filename="*q8_0.gguf"
# )
# LLM_LOAD_PARAMS = dict(
#     model_path='../llm_models/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-F16.gguf'
# )
# LLM_LOAD_PARAMS = dict(
#     model_path="../llm_models/Llama-3.2-3B-Instruct/Llama-3.2-3B-Instruct-F16.gguf"
# )

CONTEXT_DOC_TEMPLATE = """$date
----------

$text

///
"""

INSTRUCTIONS_TEMPLATE = {
    "en": f"""You are a helpful assistant.
    
The following texts are diary entries of the user that will ask you some questions. They are not sorted and start with a
date in year-month-day format, followed by "----------" and then the text entry. Each diary entry ends with "///".
Answer any questions related to the journal entries. If you cannot give an answer, only return the text
"{NO_ANSW_SENTINEL}". Here are the journal entries:

$documents
""",
    "de": f"""Du bist ein hilfreicher und höflicher Assistent, der in den Antworten siezt.

Die folgenden Texte sind Tagebucheinträge des Nutzers, der gleich einige Fragen stellen wird. Sie sind nicht sortiert
und starten mit einem Datum im Jahr-Monat-Tag Format, gefolgt von "----------" und im Anschluss dem Texteintrag. Jeder 
Tagebucheintrag endet mit "///". Beantworte alle Fragen in Bezug auf die Tagebucheinträge. Falls sich eine Frage nicht
beantworten lässt, antworte ausschließlich mit dem Text "{NO_ANSW_SENTINEL}". Hier sind die Tagebucheinträge:

$documents
""",
}

NO_ANSW_INFO = {
    "en": "I cannot give an answer to this question with the provided diary notes. Please try to rephrase your "
    "question.",
    "de": "Auf diese Frage kann ich anhand der bereitgestellten Tagebuchnotizen keine Antwort geben. Versuchen Sie bitte, Ihre Frage anders zu formulieren.",
}


def printerr(*args):
    """Shortcut to print to stdout."""
    print(*args, file=sys.stderr)


def add_message(messages, role, msg):
    """Add a message for a specific role to the chat history."""
    messages.append({"role": role, "content": msg})


def add_system_message(messages, msg):
    """Add a system message to the chat history."""
    messages.append({"role": "system", "content": msg})


def add_user_message(messages, msg):
    """Add a user message to the chat history."""
    messages.append({"role": "user", "content": msg})


def add_assistant_message(messages, msg):
    """Add an assistant message to the chat history."""
    messages.append({"role": "assistant", "content": msg})


def log_time(logger, t_start, msg=""):
    """Helper function to log time since `t_start`."""
    if msg:
        msg += " "
    logger.info(f"{msg}took {round(time() - t_start, 1)} sec.")


def load_documents(
    memexdb, num_workers, encoding, key, key_passwd, sample, use_cache, logger
):
    """
    Load documents from memex store.

    :param memexdb: memex files loaction
    :param num_workers: number of workers for parallel decryption
    :param encoding: text file encoding
    :param key: path to private GPG key
    :param key_passwd: key password, if key is locked
    :param sample: if not None, draw a sample of documents
    :param use_cache: if True, cache the decrypted documents in /tmp -- use with care!
    :param logger: logger instance
    :return: tuple (document dates list, document list, document fragments list)
    """
    memexdb_path = Path(memexdb)
    if not memexdb_path.exists():
        printerr(f'The memexdb path does not exist: "{memexdb_path}"')
        exit(1)

    memexfiles = sorted(memexdb_path.glob("*/*.tar.gpg"))
    if not memexfiles:
        printerr(f'Could not find any memex files in memexdb at "{memexdb_path}".')
        exit(1)
    logger.info(f"found {len(memexfiles)} documents")

    if use_cache:
        h = hashlib.new("sha256")
        h.update("".join(map(str, memexfiles)).encode() + str(sample).encode())
        cachefile = Path(tempfile.gettempdir()) / f"{h.hexdigest()}.pickle"
    else:
        cachefile = None

    if cachefile and cachefile.exists():
        logger.info("loading documents from cache")
        with open(cachefile, "rb") as f:
            doc_dates, docs = pickle.load(f)
    else:
        if sample:
            sample = min(sample, len(memexfiles))
            logger.info(f"will sample {sample} documents")
            memexfiles = random.sample(memexfiles, sample)

        logger.info(
            f'loading and decrypting {len(memexfiles)} documents from memexdb at "{memexdb_path}" using '
            f"{num_workers} workers"
        )

        # decrypt the files in parallel
        decrypt_memex_w_args = partial(
            decrypt_doc, encoding=encoding, key_fpath=key, key_passwd=key_passwd
        )

        t_start = time()
        with mp.Pool(num_workers) as proc_pool:
            res = proc_pool.map(decrypt_memex_w_args, memexfiles)
            doc_dates, docs = zip(*res)
        log_time(logger, t_start)

        if cachefile:
            logger.info("saving documents to cache")
            with open(cachefile, "wb") as f:
                pickle.dump((doc_dates, docs), f)

    # create document fragments for better search results simply by splitting by line
    doc_fragments = [
        [line.strip() for line in doc.split("\n") if line.strip()] for doc in docs
    ]
    logger.info(f"generated {sum(map(len, doc_fragments))} document fragments")

    return doc_dates, docs, doc_fragments


def decrypt_doc(fpath, encoding, key_fpath, key_passwd):
    """
    Decrypt an encrypted memex file.

    :param fpath: path to memex file
    :param encoding: text file encoding
    :param key_fpath: path to private GPG key
    :param key_passwd: key password, if key is locked
    :return: tuple with (date, decrypted text)
    """
    date = fpath.name.removesuffix(".tar.gpg")

    msg = pgpy.PGPMessage.from_file(fpath)
    key_loaded, _ = pgpy.PGPKey.from_file(key_fpath)

    if key_loaded.is_unlocked:
        decrypted = key_loaded.decrypt(msg).message
    else:
        with key_loaded.unlock(key_passwd) as key:
            decrypted = key.decrypt(msg).message

    tar = TarFile(fileobj=BytesIO(decrypted))
    try:
        txtbuffer = tar.extractfile(f"./{date}.txt")
    except KeyError:
        raise SystemError(
            f'could not find memex text file in tar archive inside "{fpath}"'
        )
    return date, txtbuffer.read().decode(encoding)


def process_input(
    query,
    messages,
    doc_dates,
    docs,
    doc_fragments,
    crossenc,
    llm,
    num_context_docs,
    lang,
    num_workers,
    no_answ_counter,
    logger,
):
    """
    Prompt and process user input, show LLM output.

    :param query: initial input query or None; in latter case, the user will be prompted for a query
    :param messages: chat history messages
    :param doc_dates: document dates as from `load_documents()`
    :param docs: document texts as from `load_documents()`
    :param doc_fragments: document fragments as from `load_documents()`
    :param crossenc: cross-encoder model for document retrieval
    :param llm: large lang. model for text generation
    :param num_context_docs: number of documents to provide as RAG context
    :param lang: detected document language
    :param num_workers: number of workers for parallel processing
    :param no_answ_counter: counter for "no answer" responses from the LLM
    :param logger: logger instance
    :return: tuple (continue flag, chat history messages, next query, updated counter for "no answer" responses from the LLM)
    """
    # handle input
    while not query:
        query = input(
            "Please enter your query. "
            'Enter "quit" to stop the program. '
            'Enter "flush" to reset the chat context.\n\n'
        )

        query = query.strip()
        logger.info(f"user query: {query}")

        if query.lower() == "quit":
            logger.info("exiting...")
            return False, [], None, 0
        elif query.lower() == "flush":
            query = None
            messages = []
            logger.info("cleared chat context")

    # generate initial system instructions including the RAG context if not given yet
    if not messages:
        # perform document retrieval using the cross encoder model on the document fragments
        logger.info("generating system instructions")
        print(
            f"Finding the {num_context_docs} most relevant documents out of {len(docs)} documents ..."
        )

        doc_frags_flat = []
        doc_frags_indices = []
        for doc_index, doc_frags in enumerate(doc_fragments):
            doc_frags_flat.extend(doc_frags)
            doc_frags_indices.extend([doc_index] * len(doc_frags))
        assert len(doc_frags_flat) == len(doc_frags_indices)

        logger.info(
            f"ranking {len(doc_frags_flat)} document fragments using {num_workers} workers and batch size {CROSS_ENC_BATCH_SIZE}"
        )
        t_start = time()
        ranking_res = crossenc.rank(
            query,
            doc_frags_flat,
            num_workers=num_workers,
            batch_size=CROSS_ENC_BATCH_SIZE,
            show_progress_bar=True,
        )
        log_time(logger, t_start)

        # identify the documents belonging to the ranked fragments
        context_documents = []
        context_templ = string.Template(CONTEXT_DOC_TEMPLATE)
        added_doc_indices = set()
        for rank in ranking_res:
            index_into_doc_frags = rank["corpus_id"]
            doc_index = doc_frags_indices[index_into_doc_frags]

            if doc_index not in added_doc_indices:
                ctx_doc_date = doc_dates[doc_index]
                ctx_doc_text = docs[doc_index]
                logger.info(
                    f"> adding context document from {ctx_doc_date}: "
                    f"{textwrap.shorten(ctx_doc_text, width=50, placeholder='...')}"
                )
                context_documents.append(
                    context_templ.substitute(date=ctx_doc_date, text=ctx_doc_text)
                )
                added_doc_indices.add(doc_index)
                if len(added_doc_indices) >= num_context_docs:
                    break

        # construct the instructions
        instructions_templ = string.Template(INSTRUCTIONS_TEMPLATE[lang])
        instructions = instructions_templ.substitute(
            documents="\n".join(context_documents)
        )
        add_system_message(messages, instructions)

    # add the user input
    add_user_message(messages, query)

    logger.info("generating output text for the following input messages:")
    for msg in messages:
        logger.info(
            f"> [{msg['role']}] {textwrap.shorten(msg['content'], width=50, placeholder='...')}"
        )

    # generate the LLM response
    print("Assistant answer:")
    t_start = time()
    streamer = llm.create_chat_completion(messages=messages, stream=True)
    hold_back_n_chars = len(NO_ANSW_SENTINEL)
    held_back = True
    generated_text = ""
    for output in streamer:
        if output["choices"]:
            output_chunk = output["choices"][0]["delta"]
            if output_chunk and "content" in output_chunk:
                new_text = output_chunk["content"]
                if len(generated_text) > hold_back_n_chars:
                    if held_back:
                        print(generated_text, end="")
                        held_back = False
                    print(new_text, end="")
                generated_text += new_text
                if generated_text == NO_ANSW_SENTINEL:
                    logger.info(
                        f'received "no answer" sentinel response from LLM; no answer counter is at '
                        f"{no_answ_counter}"
                    )
                    if no_answ_counter == 0:
                        return True, [], query, no_answ_counter + 1
                    else:
                        print(NO_ANSW_INFO[lang])
                        return True, [], None, 0
    else:
        add_assistant_message(messages, generated_text)
        print("\n")
        logger.info(f"generated output text of length {len(generated_text)}")
    log_time(logger, t_start)

    return True, messages, None, 0


def main():
    """Script main routine."""

    # set up and parse arguments
    argparser = ArgumentParser()
    argparser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="initial query; leave empty to be prompted",
    )
    argparser.add_argument(
        "--key",
        default=str(DEFAULT_GPG_PRIV_KEY),
        help="path to private GPG key file in ASC format",
    )
    argparser.add_argument(
        "--keypw",
        default="",
        help="private GPG key password; leave empty to be prompted when key is locked",
    )
    argparser.add_argument(
        "--memexdb", default=str(DEFAULT_MEMEXDB_PATH), help="path to memex files"
    )
    argparser.add_argument(
        "--num-context-docs",
        type=int,
        default=DEFAULT_NUM_CONTEXT_DOCS,
        help="number of memex notes that make up the RAG context",
    )
    argparser.add_argument(
        "--num-context-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help="maximum number of input tokens",
    )
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_N_WORKERS,
        help="number of worker processes used for parallel processing",
    )
    argparser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="draw only a sample of the memex notes by specifying the sample size",
    )
    argparser.add_argument(
        "--encoding", default=DEFAULT_ENCODING, help="memex notes encoding"
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="turn on verbose output"
    )
    argparser.add_argument(
        "--use-cache",
        action="store_true",
        help="cache decrypted notes in temp. directory; use with care!",
    )

    args = argparser.parse_args()

    print("Starting up the assistent ...")

    # set up logging
    logging.basicConfig()
    logger = getLogger(__name__)
    if args.verbose:
        logger.setLevel(logging.INFO)

    # set up GPG key
    logger.info(f'using GPG key "{args.key}"')
    key, _ = pgpy.PGPKey.from_file(args.key)
    if args.keypw:
        key_passwd = args.keypw
    else:
        key_passwd = ""

    if not key.is_unlocked:
        if not key_passwd:
            key_passwd = getpass(
                "The provided key is locked with a password. Please provide the password: "
            )

        try:
            with key.unlock(key_passwd) as _:
                pass
        except pgpy.errors.PGPDecryptionError:
            print("The provided key password is incorrect.")
            exit(1)

    # load cross encoder model
    logger.info(f'loading CrossEncoder model "{DEFAULT_EMBEDDINGS_MODEL}"')
    crossenc = CrossEncoder(DEFAULT_EMBEDDINGS_MODEL)

    # load language classifier model and LLM
    logger.info(f'loading language detection model "{DEFAULT_LANG_DETECT_MODEL}"')
    lang_classifier = transformers.pipeline(
        "text-classification", model=DEFAULT_LANG_DETECT_MODEL, use_fast=False
    )

    logger.info(f"loading LLM with parameters {LLM_LOAD_PARAMS}")
    llm_load_params = LLM_LOAD_PARAMS.copy()
    llm_load_params.update(dict(n_ctx=args.num_context_tokens, verbose=args.verbose))

    if "repo_id" in llm_load_params:
        llm = Llama.from_pretrained(**llm_load_params)
    else:
        llm = Llama(**llm_load_params)

    # decrypt memex documents
    print("Loading memex documents ...")
    doc_dates, docs, doc_fragments = load_documents(
        memexdb=args.memexdb,
        num_workers=args.num_workers,
        encoding=args.encoding,
        key=args.key,
        key_passwd=key_passwd,
        sample=args.sample,
        use_cache=args.use_cache,
        logger=logger,
    )

    # detect language using the first document
    lang_classif_res = lang_classifier(docs[0], top_k=1, truncation=True)
    lang = "en"
    if lang_classif_res:
        lang = lang_classif_res[0]["label"]
    print(f'Detected document language "{lang}".')

    if lang not in INSTRUCTIONS_TEMPLATE:
        printerr(f'Language not supported: "{lang}".')
        exit(1)

    # run chat loop
    cont = True
    messages = []
    query = args.query
    no_answ_counter = 0
    while cont:
        cont, messages, query, no_answ_counter = process_input(
            query,
            messages,
            doc_dates,
            docs,
            doc_fragments,
            crossenc=crossenc,
            llm=llm,
            num_context_docs=args.num_context_docs,
            lang=lang,
            num_workers=args.num_workers,
            no_answ_counter=no_answ_counter,
            logger=logger,
        )

    print("Done.")


# script entry point
if __name__ == "__main__":
    main()
