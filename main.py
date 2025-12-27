from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher

INDEX_READER = IndexReader.from_prebuilt_index('robust04')
LUCENE_SEARCHER = LuceneSearcher.from_prebuilt_index('robust04')

