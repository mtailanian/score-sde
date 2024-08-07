import itertools as it
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from mpmath import mp
from skimage.morphology import max_tree
from torch.nn import functional as F
import scipy.stats as st

mp.dps = 15
SHOW = False


def compute_nfa_anomaly_score_tree(
        z: List[torch.Tensor],
        target_size: int = 256,
        upsample_mode='bilinear',
        sigma=1.,
        reduce='median'

):

    # NFA by region (Tree)
    log_prob = []
    for img_idx in range(z[0].shape[0]):
        log_prob_s = []
        for zi in z:
            nfa_tree = NFATree(zi[img_idx], sigma, reduce)
            log_prob_s.append(nfa_tree.compute_log_prob_map())
        log_prob.append(
            torch.cat([
                F.interpolate(
                    torch.from_numpy(log_prob_s_i).nan_to_num(0).unsqueeze(0).unsqueeze(0),
                    size=(target_size, target_size),
                    mode=upsample_mode,
                    **{'align_corners': False} if 'nearest' not in upsample_mode else {}
                ) for log_prob_s_i in log_prob_s
            ], dim=1)
        )
    log_prob = torch.cat(log_prob, dim=0)

    log_prob = log_prob.amin(dim=1, keepdim=True)

    # log_n_tests = compute_number_of_tests([int(zi.shape[-1] * zi.shape[-2]) for zi in z])
    # TODO: CHECK
    log_n_tests = compute_number_of_tests([int(zi.shape[-1])])

    log_nfa = log_n_tests + log_prob

    return -log_nfa


class NFATree:
    def __init__(self, zi, sigma=1., reduce='median'):
        self.sigma = sigma
        self.reduce = reduce
        self.n_channels = zi.shape[0]
        self.zi_l2 = (zi ** 2).sum(dim=0).sqrt().cpu().numpy()
        self.zi_abs_rav = zi.abs().amax(dim=0).cpu().numpy().ravel()

        self.original_shape = self.zi_l2.shape

        self.tree = self.build_tree()

    def compute_log_prob_map(self):
        self.compute_log_prob()

        self.nfa_prune()
        keep_merging = self.nfa_merge()
        while keep_merging:
            self.nfa_prune()
            keep_merging = self.nfa_merge()
        self.nfa_prune()

        log_prob_map = np.empty(self.original_shape[0] * self.original_shape[1], dtype=np.float32)
        log_prob_map[:] = np.nan

        final_clusters = self.get_final_clusters()

        for log_prob, pixels in final_clusters.items():
            log_prob_map[pixels] = log_prob

        log_prob_map = log_prob_map.reshape(self.original_shape)

        # DEBUG
        if SHOW:
            log_nfa_map = log_prob_map + compute_number_of_tests([self.original_shape[-1]])
            plt.imshow(log_nfa_map)
            plt.show()
            plt.imshow(log_nfa_map < 0)
            plt.show()
            plt.plot(np.sort(np.unique(log_nfa_map)), '.')
            plt.grid()
            plt.show()

        return log_prob_map

    def compute_log_prob(self):
        for n in self.tree.nodes:
            region = self.tree.nodes[n]['pixels']

            zi_obs = self.zi_abs_rav[region]

            if self.reduce == 'median':
                zi_obs = np.median(zi_obs)
            elif self.reduce == 'max':
                zi_obs = np.max(zi_obs)
            elif self.reduce == 'mean':
                zi_obs = np.mean(zi_obs)
            elif self.reduce == 'min':
                zi_obs = np.min(zi_obs)
            else:
                print(f"Reduction {self.reduce} not supported. Use 'median', 'max', 'mean' or 'min'")
                raise ValueError

            log_prob = np.log10(2) + st.norm.logsf(zi_obs, scale=self.sigma) / np.log(10)

            # Log prob for the whole region
            self.tree.nodes[n]['log_prob'] = len(region) * log_prob

    def build_tree(self):
        parents, pixel_indices = max_tree(self.zi_l2, connectivity=1)
        parents_rav = parents.ravel()
        score_rav = self.zi_l2.ravel()

        tree = nx.DiGraph()
        tree.add_nodes_from(pixel_indices)
        for node in tree.nodes():
            tree.nodes[node]['score'] = score_rav[node]
        tree.add_edges_from([(n, parents_rav[n]) for n in pixel_indices[1:]])

        self.prune(tree, pixel_indices[0])
        self.accumulate(tree, pixel_indices[0])

        return tree

    def prune(self, graph, starting_node):
        """
        Transform a canonical max tree to a max tree.
        """
        value = graph.nodes[starting_node]['score']
        cluster_nodes = [starting_node]
        for p in [p for p in graph.predecessors(starting_node)]:
            if (graph.nodes[p]['score'] == value):
                cluster_nodes.append(p)
                graph.remove_node(p)
            else:
                self.prune(graph, p)
        graph.nodes[starting_node]['pixels'] = cluster_nodes
        return

    def accumulate(self, graph, starting_node):
        """
        Transform a max tree to a component tree.
        """
        pixels = graph.nodes[starting_node]['pixels']
        for p in graph.predecessors(starting_node):
            pixels.extend(self.accumulate(graph, p))
        return pixels

    def get_branch(self, starting_node):
        branch = [starting_node]
        successors = [s for s in self.tree.successors(starting_node)]

        if len(successors) == 0:
            return branch
        assert len(successors) == 1, "Node has more than one successor"

        is_only_child = len([p for p in self.tree.predecessors(successors[0])]) == 1
        if is_only_child:
            branch.extend(self.get_branch(successors[0]))
        return branch

    def get_final_clusters(self):
        leaves = [p for p in self.tree.pred if len(self.tree.pred[p]) == 0]
        final_clusters = {}
        for l in leaves:
            branch_nodes = self.get_branch(l)
            branch_log_probs = [self.tree.nodes[b]['log_prob'] for b in branch_nodes]
            branch_chosen_node = branch_nodes[np.argmin(branch_log_probs)]
            final_clusters[self.tree.nodes[branch_chosen_node]['log_prob']] = self.tree.nodes[branch_chosen_node]['pixels']
        return final_clusters

    def nfa_prune(self):
        leaves = [p for p in self.tree.pred if len(self.tree.pred[p]) == 0]
        for l in leaves:
            branch_nodes = self.get_branch(l)
            branch_log_probs = [self.tree.nodes[b]['log_prob'] for b in branch_nodes]
            chosen_node = np.argmin(branch_log_probs)
            for i in range(len(branch_nodes)):
                if i != chosen_node:
                # if i > chosen_node:
                    self.tree.add_edges_from(
                        it.product(self.tree.predecessors(branch_nodes[i]), self.tree.successors(branch_nodes[i]))
                    )
                    self.tree.remove_node(branch_nodes[i])

    def nfa_merge(self):
        merged = False
        bifurcations = [p for p in self.tree.pred if len(self.tree.pred[p]) > 1]
        for b in bifurcations:
            # if predecessors are not leaves, continue. We only merge leaves.
            if np.sum([len([pp for pp in self.tree.predecessors(p)]) for p in self.tree.predecessors(b)]) > 0:
                continue
            preds = [p for p in self.tree.predecessors(b)]
            preds_nfas = [self.tree.nodes[p]['log_prob'] for p in preds]
            if self.tree.nodes[b]['log_prob'] <= np.min(preds_nfas):  # TODO: check

            #
            #
            # big_region = self.tree.nodes[b]['log_prob'] + compute_n_test_single(len(self.tree.nodes[b]['pixels']))
            # small_regions = 0
            # for pi in preds:
            #     small_regions += self.tree.nodes[pi]['log_prob'] + compute_n_test_single(len(self.tree.nodes[pi]['pixels']))
            # if big_region <= small_regions:  # TODO: check
            #
            #

                merged = True
                for p in preds:
                    self.tree.add_edges_from(
                        it.product(self.tree.predecessors(p), self.tree.successors(p))
                    )
                    self.tree.remove_node(p)
        return merged


def compute_number_of_tests(polyominoes_sizes):
    alpha = mp.mpf(0.316915)
    beta = mp.mpf(4.062570)

    if not isinstance(polyominoes_sizes, list):
        polyominoes_sizes = [polyominoes_sizes]

    n_test = mp.mpf(0)
    for region_size in polyominoes_sizes:
        n_test_i = mp.mpf(0)
        for r in range(1, region_size + 1):
            region_size_mp = mp.mpf(r)
            n_test_i += alpha * beta ** region_size_mp / region_size_mp
        n_test += (n_test_i * region_size)

    return float(np.array(mp.log10(n_test), dtype=np.float32))


def compute_n_test_single(region_size):
    alpha = mp.mpf(0.316915)
    beta = mp.mpf(4.062570)

    n_test_i = mp.mpf(0)
    region_size_mp = mp.mpf(region_size)
    n_test_i += alpha * beta ** region_size_mp / region_size_mp

    return float(np.array(mp.log10(n_test_i), dtype=np.float32))



def debug(self):
    log_prob_map = np.empty(self.original_shape[0] * self.original_shape[1], dtype=np.float32)
    log_prob_map[:] = np.nan

    final_clusters = self.get_final_clusters()
    # final_clusters = self.grow_region(final_clusters)

    for log_prob, pixels in final_clusters.items():
        log_prob_map[pixels] = log_prob

    log_prob_map = log_prob_map.reshape(self.original_shape)

    log_nfa_map = log_prob_map + compute_number_of_tests([self.zi2_rav.shape[-1]])
    plt.imshow(log_nfa_map)
    plt.show()
    print(np.nanmin(log_nfa_map), np.nanmax(log_nfa_map))