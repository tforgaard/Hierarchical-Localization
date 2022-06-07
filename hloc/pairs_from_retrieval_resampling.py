import argparse
from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import torch
import collections.abc as collections
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from . import logger
from .utils.parsers import parse_image_lists
from .utils.read_write_model import read_images_binary
from .utils.io import list_h5_names


def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
    elif names is not None:
        if isinstance(names, (str, Path)):
            names = parse_image_lists(names)
        elif isinstance(names, collections.Iterable):
            names = list(names)
        else:
            raise ValueError(f'Unknown type of image list: {names}.'
                             'Provide either a list or a path to a list file.')
    else:
        names = names_all
    return names


def get_descriptors(names, path, name2idx=None, key='global_descriptor'):
    if name2idx is None:
        with h5py.File(str(path), 'r') as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), 'r') as fd:
                desc.append(fd[n][key].__array__())
    return torch.from_numpy(np.stack(desc, 0)).float()


def pairs_from_score_matrix(scores: torch.Tensor,
                            invalid: np.array,
                            num_select: int,
                            min_score: Optional[float] = None):
    assert scores.shape == invalid.shape
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    invalid = torch.from_numpy(invalid).to(scores.device)
    if min_score is not None:
        invalid |= scores < min_score
    scores.masked_fill_(invalid, float('-inf'))

    topk = torch.topk(scores, num_select, dim=1)
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()

    return topk.values.cpu().numpy(), indices, valid


def top_pairs_from_score_matrix(scores: torch.Tensor,
                                num_select: int,
                                min_score: Optional[float] = None):
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    if min_score is not None:
        invalid = scores < min_score
    scores.masked_fill_(invalid, float('-inf'))


    topk = torch.topk(scores.flatten(), num_select)
    indices = np.stack(np.unravel_index(topk.indices.cpu().numpy(), scores.shape))
    p = np.argsort(indices[0])
    indices[0] = indices[0][p]
    indices[1] = indices[1][p]
    # valid = topk.values.isfinite().cpu().numpy().reshape(scores.shape)

    pairs = []
    for i, j in zip(*indices):
        if scores[i,j].isfinite():
            pairs.append((i, j))
    return pairs

def main(descriptors, output, num_matched,
         query_list, query_interval=1, resample_runs=0,
         db_prefix=None, db_list=None, db_model=None, db_descriptors=None,
         match_mask=None, min_score=0, visualize=False, seed=None):
    logger.info('Extracting image pairs from a retrieval database.')

    # We handle multiple reference feature files.
    # We only assume that names are unique among them and map names to files.
    if db_descriptors is None:
        db_descriptors = descriptors
    if isinstance(db_descriptors, (Path, str)):
        db_descriptors = [db_descriptors]
    name2db = {n: i for i, p in enumerate(db_descriptors)
               for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())
    query_names_h5 = list_h5_names(descriptors)

    if db_model:
        images = read_images_binary(db_model / 'images.bin')
        db_names = [i.name for i in images.values()]
    else:
        db_names = parse_names(db_prefix, db_list, db_names_h5)
    if len(db_names) == 0:
        raise ValueError('Could not find any database image.')
    db_names = sorted(db_names)
    db_desc = get_descriptors(db_names, db_descriptors, name2db)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # RESAMPLING
    all_query_names = sorted(parse_names(None, query_list, query_names_h5))
    Q = len(all_query_names)

    if match_mask is None:
        # Avoid self-matching
        match_mask = np.array(all_query_names)[:, None] == np.array(db_names)[None]
    else:
        assert match_mask.shape == (len(all_query_names), len(db_names)), "mask shape must match size of query and database images!"

    name2idx = {name:i for i,name in enumerate(all_query_names)}

    resample_runs = min(resample_runs, query_interval-1) 

    # score matrix of size [# query images x # db images]
    scores = np.zeros((Q,len(db_names)))

    # queries for first retrieval run
    queries = query_list[::query_interval]

    # add last element for better interpolation
    if queries[-1] != query_list[-1]:
        queries.append(query_list[-1])

    num_queries = len(queries)

    for run in range(resample_runs+1):

        query_names = sorted(parse_names(None, queries, query_names_h5))
        q = len(query_names)

        query_desc = get_descriptors(query_names, descriptors)
        sim = torch.einsum('id,jd->ij', query_desc.to(device), db_desc.to(device))

        self = np.zeros((q,len(db_names)),dtype=bool)
        for i, query_name in enumerate(query_names):
            idx = name2idx[query_name]
            self[i,:] = match_mask[idx,:]
        
        values, indices, valid = pairs_from_score_matrix(sim, self, num_matched)

        for i,j in zip(*np.where(valid)):
            scores[name2idx[query_names[i]],indices[i,j]] = values[i,j]

        score_vals = scores.mean(axis=1)

        # Find scores that are actually calculated
        score_mask = score_vals != 0.0
        
        # Fix for negative scores
        score_vals[score_vals < 0.0] = 0.0
        
        # interpolate probabilites between query samples
        x = np.where(score_mask)[0]
        f = interp1d(x, score_vals[score_mask])

        xnew = np.arange(Q)
        score_vals_new = f(xnew)
        # Set already sampled queries to 0 prob
        score_vals_new[score_mask] = 0.0

        xk = np.arange(len(score_vals_new))
        if abs(sum(score_vals_new)) <= 0.000002:
            pk = np.ones_like(score_vals_new) * 1 / len(score_vals_new)
        else:
            pk = score_vals_new / sum(score_vals_new)
        custm = stats.rv_discrete(name='custm', values=(xk, pk), seed=seed)

        # Sample new query images according to already sampled query images
        R = custm.rvs(size=q)
        queries = [query_list[r] for r in R]

        # debug plotting
        if visualize:
            outputs = Path(output).parent

            plt.clf()
            plt.plot(xk, custm.pmf(xk), 'ro', ms=4, mec='r')
            plt.vlines(xk, 0, custm.pmf(xk), colors='r', lw=1)
            plt.savefig(outputs / f"probabilites_run{run}.png")

            plt.clf()
            plt.imshow(scores, interpolation=None)
            plt.savefig(outputs / f"scores_run{run}.png")

    pairs = top_pairs_from_score_matrix(scores,num_select=num_matched*num_queries, min_score=min_score)

    scores_final = np.zeros((Q,len(db_names)))
    for i, j in pairs:
        scores_final[i,j] = scores[i,j]
    scores_final_masked = scores_final[scores_final>0.0]

    score_final = 0.0
    if len(scores_final_masked):
        score_final = np.percentile(scores_final_masked,90)

    pairs = [(all_query_names[i], db_names[j]) for i, j in pairs if scores_final[i,j] > 0.0]

    logger.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))

    if visualize:
        # Final sample distribution
        xk = np.arange(Q)
        pk = score_vals / sum(score_vals)
        custm = stats.rv_discrete(name='custm', values=(xk, pk))

        plt.clf()
        plt.plot(xk, custm.pmf(xk), 'ro', ms=4, mec='r')
        plt.vlines(xk, 0, custm.pmf(xk), colors='r', lw=1)
        plt.savefig(outputs / "probabilites_final.png")

        plt.clf()
        plt.imshow(scores_final, interpolation=None)
        plt.savefig(outputs / "scores_final.png")

    return [q for q, _ in pairs], score_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_matched', type=int, required=True)
    parser.add_argument('--query_prefix', type=str, nargs='+')
    parser.add_argument('--query_list', type=Path)
    parser.add_argument('--db_prefix', type=str, nargs='+')
    parser.add_argument('--db_list', type=Path)
    parser.add_argument('--db_model', type=Path)
    parser.add_argument('--db_descriptors', type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
