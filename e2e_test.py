import os

from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig, RunConfig, Run
from colbert.data import Queries

CKPT = "/dfs/scratch0/okhattab/share/2021/checkpoints/msmarco.psg.kldR2.nway64.ib__colbert-400000"
STACKEXCHANGE_DATA_PATH = "/future/u/xmgui/ColBERT/docs/downloads/lotte"
EXPERIMENT_DIR = "/future/u/xmgui/ColBERT/experiments"

STACKEXCHANGE_DATASETS = [
    "lifestyle",
]

def evaluate(dataset, dataset_path):
    split = "dev"
    nbits = 2  # encode each dimension with 2 bits
    doc_maxlen = 300  # truncate passages at 300 tokens
    collection = os.path.join(dataset_path, dataset, split, 'collection.tsv')
    index_name = f'{dataset}.{split}.{nbits}bits.latest'
    experiment = (f"{dataset}.{split}.nbits={nbits}",)

    with Run().context(RunConfig(nranks=4)):
        config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            nbits=nbits,
            root=EXPERIMENT_DIR,
            experiment=experiment,
        )
        indexer = Indexer(CKPT, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)

        config = ColBERTConfig(
            root=EXPERIMENT_DIR,
            experiment=experiment,
            nprobe=2,
            ncandidates=8192,
        )
        searcher = Searcher(
            index=index_name,
            config=config,
        )
        for query_type in ["search", "forum"]:
            queries = os.path.join(
                dataset_path, dataset, split, f"questions.{query_type}.tsv"
            )
            queries = Queries(path=queries)
            ranking = searcher.search_all(queries, k=5)
            ranking.save(f"{dataset}.{query_type}.ranking.tsv")


def main():
    for dataset in STACKEXCHANGE_DATASETS:
        evaluate(dataset, STACKEXCHANGE_DATA_PATH)


if __name__ == "__main__":
    main()