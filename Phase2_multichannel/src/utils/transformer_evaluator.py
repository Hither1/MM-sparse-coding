import json
import os
import pickle
import time
from collections import defaultdict

import numba
import numpy as np
import torch
from tqdm.auto import tqdm

from .inverted_index import IndexDictOfArray


def to_list(tensor):
    return tensor.detach().cpu().tolist()


class L0:
    """non-differentiable
    """

    def __call__(self, batch_rep):
        return torch.count_nonzero(batch_rep, dim=-1).float().mean()


class Evaluator:
    def __init__(self, model, config=None, restore=True):
        """
        :param model: model
        :param config: config dict
        :param restore: restore model true by default
        """
        self.model = model
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if restore:  
            if self.device == torch.device("cuda"):
                if 'hf_training'  in config:
                    ## model already loaded
                    pass

                self.model.eval()
                if torch.cuda.device_count() > 1:
                    print(" --- use {} GPUs --- ".format(torch.cuda.device_count()))
                    self.model = torch.nn.DataParallel(self.model)
                self.model.to(self.device)
#                    print("restore model on GPU at {}".format(os.path.join(config["checkpoint_dir"], "model")),
#                        flush=True)
            else:  # CPU
                if 'hf_training'  in config:
                    ## model already loaded
                    pass                    
        else:
            print("WARNING: init evaluator, NOT restoring the model, NOT placing on device")
        self.model.eval()  # => put in eval mode


class SparseIndexing(Evaluator):
    """sparse indexing
    """

    def __init__(self, model, config, compute_stats=False, dim_voc=None, is_query=False, force_new=True,**kwargs):
        super().__init__(model, config, **kwargs)
        self.index_dir = config["index_dir"] if config is not None else None
        self.sparse_index = IndexDictOfArray(self.index_dir, dim_voc=dim_voc, force_new=force_new)
        self.compute_stats = compute_stats
        self.is_query = is_query
        if self.compute_stats:
            self.l0 = L0()

    def index(self, collection_loader, id_dict=None):
        doc_ids = []
        if self.compute_stats:
            stats = defaultdict(float)
        count = 0
        with torch.no_grad():
            for t, batch in enumerate(tqdm(collection_loader)):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"id"}}
                if self.is_query:
                    batch_documents = self.model.encode_text(**inputs)[1]
                else:
                    batch_documents = self.model.encode_text(**inputs)[1]
                if self.compute_stats:
                    stats["L0_d"] += self.l0(batch_documents).item()
                row, col = torch.nonzero(batch_documents, as_tuple=True)
                data = batch_documents[row, col]
                row = row + count
                batch_ids = to_list(batch["id"])
                if id_dict:
                    batch_ids = [id_dict[x] for x in batch_ids]
                count += len(batch_ids)
                doc_ids.extend(batch_ids)
                self.sparse_index.add_batch_document(row.cpu().numpy(), col.cpu().numpy(), data.cpu().numpy(),
                                                     n_docs=len(batch_ids))
        if self.compute_stats:
            stats = {key: value / len(collection_loader) for key, value in stats.items()}
        if self.index_dir is not None:
            self.sparse_index.save()
            pickle.dump(doc_ids, open(os.path.join(self.index_dir, "doc_ids.pkl"), "wb"))
            print("done iterating over the corpus...")
            print("index contains {} posting lists".format(len(self.sparse_index)))
            print("index contains {} documents".format(len(doc_ids)))
            if self.compute_stats:
                with open(os.path.join(self.index_dir, "index_stats.json"), "w") as handler:
                    json.dump(stats, handler)
        else:
            # if no index_dir, we do not write the index to disk but return it
            for key in list(self.sparse_index.index_doc_id.keys()):
                # convert to numpy
                self.sparse_index.index_doc_id[key] = np.array(self.sparse_index.index_doc_id[key], dtype=np.int32)
                self.sparse_index.index_doc_value[key] = np.array(self.sparse_index.index_doc_value[key],
                                                                  dtype=np.float32)
            out = {"index": self.sparse_index, "ids_mapping": doc_ids}
            if self.compute_stats:
                out["stats"] = stats
            return out


class SparseRetrieval(Evaluator):
    """retrieval from SparseIndexing
    """

    @staticmethod
    def select_topk(filtered_indexes, scores, k):
        if len(filtered_indexes) > k:
            sorted_ = np.argpartition(scores, k)[:k]
            filtered_indexes, scores = filtered_indexes[sorted_], -scores[sorted_]
        else:
            scores = -scores
        return filtered_indexes, scores

    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def numba_score_float(inverted_index_ids: numba.typed.Dict,
                          inverted_index_floats: numba.typed.Dict,
                          indexes_to_retrieve: np.ndarray,
                          query_values: np.ndarray,
                          threshold: float,
                          size_collection: int):
        scores = np.zeros(size_collection, dtype=np.float32)  # initialize array with size = size of collection
        n = len(indexes_to_retrieve)
        for _idx in range(n):
            local_idx = indexes_to_retrieve[_idx]  # which posting list to search
            query_float = query_values[_idx]  # what is the value of the query for this posting list
            retrieved_indexes = inverted_index_ids[local_idx]  # get indexes from posting list
            retrieved_floats = inverted_index_floats[local_idx]  # get values from posting list
            for j in numba.prange(len(retrieved_indexes)):
                scores[retrieved_indexes[j]] += query_float * retrieved_floats[j]
        filtered_indexes = np.argwhere(scores > threshold)[:, 0]  # ideally we should have a threshold to filter
        # unused documents => this should be tuned, currently it is set to 0
        return filtered_indexes, -scores[filtered_indexes]

    def __init__(self, model, config, dim_voc, dataset_name=None, index_d=None, compute_stats=False, is_beir=False,
                 **kwargs):
        super().__init__(model, config, **kwargs)
        assert ("index_dir" in config and index_d is None) or (
                "index_dir" not in config and index_d is not None)
        if "index_dir" in config:
            self.sparse_index = IndexDictOfArray(config["index_dir"], dim_voc=dim_voc)
            self.doc_ids = pickle.load(open(os.path.join(config["index_dir"], "doc_ids.pkl"), "rb"))
        else:
            self.sparse_index = index_d["index"]
            self.doc_ids = index_d["ids_mapping"]
            for i in range(dim_voc):
                # missing keys (== posting lists), causing issues for retrieval => fill with empty
                if i not in self.sparse_index.index_doc_id:
                    self.sparse_index.index_doc_id[i] = np.array([], dtype=np.int32)
                    self.sparse_index.index_doc_value[i] = np.array([], dtype=np.float32)
        # convert to numba
        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()
        for key, value in self.sparse_index.index_doc_id.items():
            self.numba_index_doc_ids[key] = value
        for key, value in self.sparse_index.index_doc_value.items():
            self.numba_index_doc_values[key] = value
        self.out_dir = os.path.join(config["out_dir"], dataset_name) if (dataset_name is not None and not is_beir) \
            else config["out_dir"]
        self.doc_stats = index_d["stats"] if (index_d is not None and compute_stats) else None
        self.compute_stats = compute_stats
        if self.compute_stats:
            self.l0 = L0()

    def retrieve(self, q_loader, top_k, name=None, return_d=False, id_dict=False, threshold=0):
        os.makedirs(self.out_dir, exist_ok=True)
        if self.compute_stats:
            os.makedirs(os.path.join(self.out_dir, "stats"), exist_ok=True)

        res = defaultdict(dict)
        if self.compute_stats:
            stats = defaultdict(float)
        with torch.no_grad():
            for t, batch in enumerate(tqdm(q_loader)):
                q_id = to_list(batch["id"])[0]
                if id_dict:
                    q_id = id_dict[q_id]
                inputs = {k: v for k, v in batch.items() if k not in {"id"}}
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                query = self.model(q_kwargs=inputs)["q_rep"]  # we assume ONE query per batch here
                if self.compute_stats:
                    stats["L0_q"] += self.l0(query).item()
                # TODO: batched version for retrieval
                row, col = torch.nonzero(query, as_tuple=True)
                values = query[to_list(row), to_list(col)]
                filtered_indexes, scores = self.numba_score_float(self.numba_index_doc_ids,
                                                                  self.numba_index_doc_values,
                                                                  col.cpu().numpy(),
                                                                  values.cpu().numpy().astype(np.float32),
                                                                  threshold=threshold,
                                                                  size_collection=self.sparse_index.nb_docs())
                # threshold set to 0 by default, could be better
                filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=top_k)
                for id_, sc in zip(filtered_indexes, scores):
                    res[str(q_id)][str(self.doc_ids[id_])] = float(sc)
        if self.compute_stats:
            stats = {key: value / len(q_loader) for key, value in stats.items()}
        if self.compute_stats:
            with open(os.path.join(self.out_dir, "stats",
                                   "q_stats{}.json".format("_iter_{}".format(name) if name is not None else "")),
                      "w") as handler:
                json.dump(stats, handler)
            if self.doc_stats is not None:
                with open(os.path.join(self.out_dir, "stats",
                                       "d_stats{}.json".format("_iter_{}".format(name) if name is not None else "")),
                          "w") as handler:
                    json.dump(self.doc_stats, handler)
        with open(os.path.join(self.out_dir, "run{}.json".format("_iter_{}".format(name) if name is not None else "")),
                  "w") as handler:
            json.dump(res, handler)
        if return_d:
            out = {"retrieval": res}
            if self.compute_stats:
                out["stats"] = stats if self.doc_stats is None else {**stats, **self.doc_stats}
            return out



class RerankEvaluator(Evaluator):

    def __init__(self, model, config, dataset_name=None, **kwargs):
        super().__init__(model, config, **kwargs)
        self.init_(config=config,dataset_name=dataset_name)

    def init_(self, config,dataset_name):
        self.out_dir = os.path.join(config["out_dir"], dataset_name) if dataset_name is not None else config["out_dir"]

    def evaluate(self, data_loader, out_dir, reranker_type, model_name="unicamp-dl/mt5-13b-mmarco-100k"):
        os.makedirs(out_dir, exist_ok=True)
        temp_d = defaultdict(dict)
        logs = open(os.path.join(out_dir, "eval_logs.txt"), "w")
        logs.write("begin evaluation on {} batches ...\n".format(len(data_loader)))
        t0 = time.time()
        with torch.no_grad():  # the model has already been put in eval mode at init
            if reranker_type == "monoT5" or reranker_type == "duoT5":
                if reranker_type == "duoT5":
                    model = DuoT5(model=self.model)
                else:
                    model = MonoT5(model=self.model,use_amp=False,pretrained_model_name_or_path=model_name)
#                    model = MonoT5(model=self.model,use_amp=False,pretrained_model_name_or_path="unicamp-dl/mt5-13b-mmarco-100k")
#                    model = MonoT5(model=self.model,use_amp=False,pretrained_model_name_or_path="castorini/monot5-3b-msmarco-10k")

                for query_id, all_list in tqdm(data_loader):
                    query = all_list[0]
                    texts = all_list[1:]
                    reranked = model.rerank(query, texts)
                    for doc in reranked:
                        temp_d[str(query_id)][str(doc.metadata["docid"])] = doc.score
            else:
                for i, batch in enumerate(tqdm(data_loader)):
                    for k, v in batch.items():
                        if k not in ["q_id", "d_id"]:
                            batch[k] = v.to(self.device)
                    logits = self.model(**{k: v for k, v in batch.items() if k not in {"q_id", "d_id","labels_attention"}})
                    logits = logits[0][:, 0]

                    for q_id, d_id, s in zip(batch["q_id"],
                                            batch["d_id"],
                                            to_list(logits),
                                            ):
                        temp_d[str(q_id)][str(d_id)] = s
        with open(os.path.join(self.out_dir, "run.json"), "w") as handler:
            json.dump(temp_d, handler)
        logs.write("done\ntook about {} hours".format((time.time() - t0) / 3600))
        return temp_d
    
  