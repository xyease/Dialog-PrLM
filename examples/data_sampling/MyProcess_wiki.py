import sys
import os
import six

from gensim.corpora import WikiCorpus
from gensim.corpora.wikicorpus import *
# parent = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(parent + '/../venv/lib/python2.7/site-packages/gensim/corpora/')


def tokenize(content):
    # override original method in wikicorpus.py
    # print(content)
    result = []
    for para in content.split("\n"):
        paralist = []
        for token in para.split():
            if len(token) <= 15 and not token.startswith('_'):
                paralist.append(token.encode('utf8'))
        if len(paralist) > 0:
            result.append(paralist)
    return result

    # return [token.encode('utf8') for para in content.split("\n") for token in para.split()
    #         if len(token) <= 15 and not token.startswith('_')]

def process_article(args):
   # override original method in wikicorpus.py
    text, lemmatize, title, pageid = args
    text = filter_wiki(text)
    if lemmatize:
        result = utils.lemmatize(text)
    else:
        result = tokenize(text)
    # print(title)
    return result, title, pageid


class MyWikiCorpus(WikiCorpus):
    def __init__(self, fname, processes=None, lemmatize=utils.has_pattern(), dictionary=None, filter_namespaces=('0',)):
        WikiCorpus.__init__(self, fname, processes, lemmatize, dictionary, filter_namespaces)
        self.metadata = True

    def get_texts(self):
        articles, articles_all = 0, 0
        positions, positions_all = 0, 0
        texts = ((text, self.lemmatize, title, pageid) for title, text, pageid in extract_pages(bz2.BZ2File(self.fname), self.filter_namespaces))
        pool = multiprocessing.Pool(self.processes)
        # process the corpus in smaller chunks of docs, because multiprocessing.Pool
        # is dumb and would load the entire input into RAM at once...
        for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
            for tokens, title, pageid in pool.imap(process_article, group):  # chunksize=10):
                articles_all += 1
                positions_all += len(tokens)
                # article redirects and short stubs are pruned here
                if len(tokens) < ARTICLE_MIN_WORDS or any(title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                    continue
                articles += 1
                positions += len(tokens)
                if self.metadata:
                    yield (tokens, (pageid, title))
                else:
                    yield tokens
        pool.terminate()

        logger.info(
            "finished iterating over Wikipedia corpus of %i documents with %i positions"
            " (total %i articles, %i positions before pruning articles shorter than %i words)",
            articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS)
        self.length = articles  # cache corpus length


def process():
    # logger = logging.getLogger(__name__)
    # logger.setLevel(level=logging.INFO)
    # logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    # # console = logging.StreamHandler()
    # # console.setLevel(logging.INFO)
    # logger.addHandler(console)

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    # if len(sys.argv) != 3:
    #     print("Using: python process_wiki.py enwiki-20180601-pages-articles.xml.bz2 wiki.en.text")
    #     sys.exit(1)
    inp, outp = "mydata/enwiki-latest-pages-articles.xml.bz2", "mydata/wikitext/"
    space = " "
    i = 0

    wiki = MyWikiCorpus(inp, lemmatize=False, dictionary={})
    # print(six.PY3)
    for (text, (pageid, title)) in wiki.get_texts():
        outputfile = os.path.join(outp, str(pageid)+".txt")
        output = open(outputfile, 'w')
        # print(title)
        # if six.PY3:
        #     mytext = ""
        #     for para in text:
        #         mypara = bytes(' '.join(para), 'utf-8').decode('utf-8')
        #         mypara += '\n'
        #         mytext += mypara
        #
        #     mystr = bytes(' '.join(title), 'utf-8').decode('utf-8') + '\n' +  mytext
        # else:
        mytext = ""
        for para in text:
            mypara = space.join([str(token, 'utf-8') for token in para])
            mypara += '\n'
            mytext += mypara
        mystr = "Title: " + title + "\n" + mytext

        i = i + 1
        # if (i % 10000 == 0):
        logger.info("Saved " + str(i) + " articles")
        output.write(mystr)
        output.close()
        # print(mystr)
        # if(i==3):
        #     break
    logger.info("Finished Saved " + str(i) + " articles")


if __name__ == '__main__':
    process()