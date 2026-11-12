from typing import List, FrozenSet, Dict, Tuple, Union, Optional
from dataclasses import dataclass

try:
    from ldpc import BpDecoder

    def create_decoder(pcm, priors, **kwargs):
        return BpDecoder(pcm=pcm, error_channel=list(priors), **kwargs)
except ImportError:
    from ldpc import bp_decoder

    def create_decoder(pcm, priors, **kwargs):
        return bp_decoder(parity_check_matrix=pcm, channel_probs=priors, **kwargs)

from scipy.sparse import csc_matrix
import numpy as np

import inspect

import stim
import pymatching

# Unreleased pymatching feature (https://github.com/oscarhiggott/PyMatching/issues/172) for temporarily
# reweighting edges of an existing graph for a single shot.
_PYMATCHING_SUPPORTS_EDGE_REWEIGHTS = (
    "edge_reweights" in inspect.signature(pymatching.Matching.decode).parameters
)


def iter_set_xor(set_list: List[List[int]]) -> FrozenSet[int]:
    out = set()
    for x in set_list:
        s = set(x)
        out = (out - s) | (s - out)
    return frozenset(out)


def dict_to_csc_matrix(
    elements_dict: Dict[int, FrozenSet[int]], shape: Tuple[int, int]
) -> csc_matrix:
    """
    Constructs a `scipy.sparse.csc_matrix` check matrix from a dictionary `elements_dict` giving the indices of nonzero
    rows in each column.

    Parameters
    ----------
    elements_dict : dict[int, frozenset[int]]
        A dictionary giving the indices of nonzero rows in each column. `elements_dict[i]` is a frozenset of ints
        giving the indices of nonzero rows in column `i`.
    shape : Tuple[int, int]
        The dimensions of the matrix to be returned

    Returns
    -------
    scipy.sparse.csc_matrix
        The `scipy.sparse.csc_matrix` check matrix defined by `elements_dict` and `shape`
    """
    nnz = sum(len(v) for k, v in elements_dict.items())
    data = np.ones(nnz, dtype=np.uint8)
    row_ind = np.zeros(nnz, dtype=np.int64)
    col_ind = np.zeros(nnz, dtype=np.int64)
    i = 0
    for col, v in elements_dict.items():
        for row in v:
            row_ind[i] = row
            col_ind[i] = col
            i += 1
    return csc_matrix((data, (row_ind, col_ind)), shape=shape)


@dataclass
class DemMatrices:
    check_matrix: csc_matrix
    observables_matrix: csc_matrix
    edge_check_matrix: csc_matrix
    edge_observables_matrix: csc_matrix
    hyperedge_to_edge_matrix: csc_matrix
    priors: np.ndarray


def detector_error_model_to_check_matrices(
    dem: stim.DetectorErrorModel, allow_undecomposed_hyperedges: bool = False
) -> DemMatrices:
    """
    Convert a `stim.DetectorErrorModel` into a `DemMatrices` object.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        A stim DetectorErrorModel
    allow_undecomposed_hyperedges: bool
        If True, don't raise an exception if a hyperedge is not decomposable. Instead, the hyperedge `h` is still added
        to the `DemMatrices.check_matrix`, `DemMatrices.observables_matrix` and `DemMatrices.priors` but it will not
        have any edges in its decomposition in `DemMatrices.hyperedge_to_edge_matrix[:, h]`.
    Returns
    -------
    DemMatrices
        A collection of matrices representing the stim DetectorErrorModel
    """
    hyperedge_ids: Dict[FrozenSet[int], int] = {}
    edge_ids: Dict[FrozenSet[int], int] = {}
    hyperedge_obs_map: Dict[int, FrozenSet[int]] = {}
    edge_obs_map: Dict[int, FrozenSet[int]] = {}
    priors_dict: Dict[int, float] = {}
    hyperedge_to_edge: Dict[int, FrozenSet[int]] = {}

    def handle_error(
        prob: float, detectors: List[List[int]], observables: List[List[int]]
    ) -> None:
        hyperedge_dets = iter_set_xor(detectors)
        hyperedge_obs = iter_set_xor(observables)

        if hyperedge_dets not in hyperedge_ids:
            hyperedge_ids[hyperedge_dets] = len(hyperedge_ids)
            priors_dict[hyperedge_ids[hyperedge_dets]] = 0.0
        hid = hyperedge_ids[hyperedge_dets]
        hyperedge_obs_map[hid] = hyperedge_obs
        priors_dict[hid] = priors_dict[hid] * (1 - prob) + prob * (1 - priors_dict[hid])

        eids = []
        for i in range(len(detectors)):
            e_dets = frozenset(detectors[i])
            e_obs = frozenset(observables[i])

            if len(e_dets) > 2:
                if not allow_undecomposed_hyperedges:
                    raise ValueError(
                        "A hyperedge error mechanism was found that was not decomposed into edges. "
                        "This can happen if you do not set `decompose_errors=True` as required when "
                        "calling `circuit.detector_error_model`."
                    )
                else:
                    continue

            if e_dets not in edge_ids:
                edge_ids[e_dets] = len(edge_ids)
            eid = edge_ids[e_dets]
            eids.append(eid)
            edge_obs_map[eid] = e_obs

        if hid not in hyperedge_to_edge:
            hyperedge_to_edge[hid] = frozenset(eids)

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets: List[List[int]] = [[]]
            frames: List[List[int]] = [[]]
            t: stim.DemTarget
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets[-1].append(t.val)
                elif t.is_logical_observable_id():
                    frames[-1].append(t.val)
                elif t.is_separator():
                    dets.append([])
                    frames.append([])
            handle_error(p, dets, frames)
        elif instruction.type == "detector":
            pass
        elif instruction.type == "logical_observable":
            pass
        else:
            raise NotImplementedError()
    check_matrix = dict_to_csc_matrix(
        {v: k for k, v in hyperedge_ids.items()},
        shape=(dem.num_detectors, len(hyperedge_ids)),
    )
    observables_matrix = dict_to_csc_matrix(
        hyperedge_obs_map, shape=(dem.num_observables, len(hyperedge_ids))
    )
    priors = np.zeros(len(hyperedge_ids))
    for i, p in priors_dict.items():
        priors[i] = p
    hyperedge_to_edge_matrix = dict_to_csc_matrix(
        hyperedge_to_edge, shape=(len(edge_ids), len(hyperedge_ids))
    )
    edge_check_matrix = dict_to_csc_matrix(
        {v: k for k, v in edge_ids.items()}, shape=(dem.num_detectors, len(edge_ids))
    )
    edge_observables_matrix = dict_to_csc_matrix(
        edge_obs_map, shape=(dem.num_observables, len(edge_ids))
    )
    return DemMatrices(
        check_matrix=check_matrix,
        observables_matrix=observables_matrix,
        edge_check_matrix=edge_check_matrix,
        edge_observables_matrix=edge_observables_matrix,
        hyperedge_to_edge_matrix=hyperedge_to_edge_matrix,
        priors=priors,
    )


class BeliefMatching:
    def __init__(
        self,
        model: Union[stim.Circuit, stim.DetectorErrorModel],
        max_bp_iters: int = 20,
        bp_method: Optional[str] = None,
        bp_backend: str = "ldpc",
        **kwargs,
    ):
        """
        Construct a BeliefMatching object from a `stim.Circuit` or `stim.DetectorErrorModel`

        Parameters
        ----------
        model : stim.Circuit or stim.DetectorErrorModel
            A stim.Circuit or a stim.DetectorErrorModel. If a stim.Circuit is provided, it will be converted
            into a stim.DetectorErrorModel using `stim.Circuit.detector_error_model(decompose_errors=True)`.
            If a `stim.DetectorErrorModel` is provided it is important that its hyperedges are decomposed
            into edges (using `decompose_errors=True`) for BeliefMatching to provide improved accuracy over
            a standard (faster) MWPM decoder.
        max_bp_iters : int
            The maximum number of iterations of belief-propagation to use. Default 20
        bp_method : str, optional
            The method of belief-propagation to use. With `bp_backend="ldpc"` this is passed to
            `ldpc.BpDecoder` as the `bp_method` argument; options are "product_sum" (alias "ps") and
            "minimum_sum" (aliases "min_sum", "ms"). The default is "product_sum". The "relay" backend only
            supports min-sum, so with it this must be None or a min-sum alias.
        bp_backend : str
            The belief-propagation implementation to use:

            - "ldpc" (default): `ldpc.BpDecoder` (https://github.com/quantumgizmos/ldpc).
            - "relay": the Rust min-sum BP decoder `relay_bp.MinSumBPDecoderF64` from relay-bp
              (https://github.com/trmue/relay). Requires `pip install relay-bp`.
        kwargs
            Additional keyword arguments are passed to the BP decoder's constructor (`ldpc.BpDecoder` or
            `relay_bp.MinSumBPDecoderF64`). The min-sum scaling factor can be given as `ms_scaling_factor`
            for either backend (for "relay" it is passed on as `alpha`). For both backends, a value in (0, 1]
            scales the check-to-variable messages by that constant, and 0 uses the adaptive schedule
            1 - 2^-(iteration + 1).
        """
        if isinstance(model, stim.Circuit):
            model = model.detector_error_model(decompose_errors=True)
        self._initialise_from_detector_error_model(
            model=model, max_bp_iters=max_bp_iters,
            bp_method=bp_method,
            bp_backend=bp_backend,
            **kwargs,
        )

    def _initialise_from_detector_error_model(
        self,
        model: stim.DetectorErrorModel,
        *,
        max_bp_iters: int = 20,
        bp_method: Optional[str] = None,
        bp_backend: str = "ldpc",
        **kwargs,
    ):
        self._model = model
        self._matrices = detector_error_model_to_check_matrices(self._model)
        self._bp_backend = bp_backend
        if bp_backend == "ldpc":
            self._bpd = create_decoder(
                pcm=self._matrices.check_matrix,
                priors=self._matrices.priors,
                max_iter=max_bp_iters,
                bp_method="product_sum" if bp_method is None else bp_method,
                input_vector_type="syndrome",
                **kwargs,
            )
        elif bp_backend == "relay":
            if bp_method not in (None, "minimum_sum", "min_sum", "ms"):
                raise ValueError(
                    f"bp_backend='relay' only supports min-sum BP, but bp_method={bp_method!r} was given."
                )
            try:
                import relay_bp
            except ImportError as e:
                raise ImportError(
                    "bp_backend='relay' requires the relay-bp package: `pip install relay-bp`."
                ) from e
            if "ms_scaling_factor" in kwargs:
                if "alpha" in kwargs:
                    raise ValueError("Give only one of `ms_scaling_factor` and `alpha`.")
                kwargs["alpha"] = float(kwargs.pop("ms_scaling_factor"))
            self._bpd = relay_bp.MinSumBPDecoderF64(
                self._matrices.check_matrix.astype(np.uint8),
                error_priors=np.asarray(self._matrices.priors, dtype=np.float64),
                max_iter=max_bp_iters,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown bp_backend {bp_backend!r}. Options are 'ldpc' and 'relay'."
            )
        self._setup_edge_reweighting()

    def _setup_edge_reweighting(self) -> None:
        """If the installed pymatching supports `edge_reweights` in `Matching.decode`, build the matching
        graph once (weighted by the prior edge probabilities) so that each shot only needs to temporarily
        reweight its edges, instead of constructing a new `Matching` object per shot."""
        self._base_matching = None
        self._edge_reweights = None
        if not _PYMATCHING_SUPPORTS_EDGE_REWEIGHTS:
            return
        eps = 1e-14
        edge_priors = np.clip(
            self._matrices.hyperedge_to_edge_matrix @ self._matrices.priors, eps, 1 - eps
        )
        base_weights = -np.log(edge_priors)
        # pymatching does not allow reweighting to change an edge's sign, and discretises new weights using
        # the base graph's scale, which leaves headroom for weights up to ~64x the largest base weight. The
        # largest weight `decode` can produce is -log(eps) ~= 32.2.
        if base_weights.min() < 0 or base_weights.max() * 32 < -np.log(eps):
            return
        h = self._matrices.edge_check_matrix.tocsc()
        if np.any(np.diff(h.indptr) > 2) or np.any(np.diff(h.indptr) == 0):
            return
        edge_reweights = np.full((h.shape[1], 3), -1.0, dtype=np.float64)
        for j in range(h.shape[1]):
            rows = h.indices[h.indptr[j] : h.indptr[j + 1]]
            edge_reweights[j, : len(rows)] = rows
        self._base_matching = pymatching.Matching.from_check_matrix(
            h,
            weights=base_weights,
            faults_matrix=self._matrices.edge_observables_matrix,
            use_virtual_boundary_node=True,
        )
        self._edge_reweights = edge_reweights

    def _run_bp(self, syndrome: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Run BP on a single syndrome. Returns (converged, correction, log-likelihood ratios), where
        the LLRs are log(P(no error) / P(error)) for each column of the check matrix."""
        if self._bp_backend == "relay":
            res = self._bpd.decode_detailed(np.ascontiguousarray(syndrome, dtype=np.uint8))
            return res.success, res.decoding, res.posterior_ratios
        corr = self._bpd.decode(syndrome)
        return bool(self._bpd.converge), corr, self._bpd.log_prob_ratios

    @classmethod
    def from_detector_error_model(
        cls,
        model: stim.DetectorErrorModel,
        *,
        max_bp_iters: int = 20,
        bp_method: Optional[str] = None,
        bp_backend: str = "ldpc",
        **kwargs,
    ) -> "BeliefMatching":
        """
        Construct a BeliefMatching object from a `stim.DetectorErrorModel`

        Parameters
        ----------
        model : stim.DetectorErrorModel
            A `stim.DetectorErrorModel`. It is important that the hyperedges are already decomposed
            into edges (using `decompose_errors=True`) for BeliefMatching to provide improved accuracy over
            a standard (faster) MWPM decoder.
        max_bp_iters : int
            The maximum number of iterations of belief-propagation to use. Default 20
        bp_method : str, optional
            The method of belief-propagation to use. With `bp_backend="ldpc"` this is passed to
            `ldpc.BpDecoder` as the `bp_method` argument; options are "product_sum" (alias "ps") and
            "minimum_sum" (aliases "min_sum", "ms"). The default is "product_sum". The "relay" backend only
            supports min-sum, so with it this must be None or a min-sum alias.
        bp_backend : str
            The belief-propagation implementation to use:

            - "ldpc" (default): `ldpc.BpDecoder` (https://github.com/quantumgizmos/ldpc).
            - "relay": the Rust min-sum BP decoder `relay_bp.MinSumBPDecoderF64` from relay-bp
              (https://github.com/trmue/relay). Requires `pip install relay-bp`.
        kwargs
            Additional keyword arguments are passed to the BP decoder's constructor (`ldpc.BpDecoder` or
            `relay_bp.MinSumBPDecoderF64`). The min-sum scaling factor can be given as `ms_scaling_factor`
            for either backend (for "relay" it is passed on as `alpha`). For both backends, a value in (0, 1]
            scales the check-to-variable messages by that constant, and 0 uses the adaptive schedule
            1 - 2^-(iteration + 1).


        Returns
        -------
        BeliefMatching
            The BeliefMatching object for decoding using `model`
        """
        bm = cls.__new__(cls)
        bm._initialise_from_detector_error_model(
            model=model, max_bp_iters=max_bp_iters,
            bp_method=bp_method,
            bp_backend=bp_backend,
            **kwargs,
        )
        return bm

    @classmethod
    def from_stim_circuit(
        cls,
        circuit: stim.Circuit,
        *,
        max_bp_iters: int = 20,
        bp_method: Optional[str] = None,
        bp_backend: str = "ldpc",
        **kwargs,
    ) -> "BeliefMatching":
        """
        Construct a BeliefMatching object from a `stim.Circuit`

        Parameters
        ----------
        circuit : stim.Circuit
            A stim.Circuit. The circuit will be converted into a stim.DetectorErrorModel using
            `stim.Circuit.detector_error_model(decompose_errors=True)`.
        max_bp_iters : int
            The maximum number of iterations of belief-propagation to use. Default 20
        bp_method : str, optional
            The method of belief-propagation to use. With `bp_backend="ldpc"` this is passed to
            `ldpc.BpDecoder` as the `bp_method` argument; options are "product_sum" (alias "ps") and
            "minimum_sum" (aliases "min_sum", "ms"). The default is "product_sum". The "relay" backend only
            supports min-sum, so with it this must be None or a min-sum alias.
        bp_backend : str
            The belief-propagation implementation to use:

            - "ldpc" (default): `ldpc.BpDecoder` (https://github.com/quantumgizmos/ldpc).
            - "relay": the Rust min-sum BP decoder `relay_bp.MinSumBPDecoderF64` from relay-bp
              (https://github.com/trmue/relay). Requires `pip install relay-bp`.
        kwargs
            Additional keyword arguments are passed to the BP decoder's constructor (`ldpc.BpDecoder` or
            `relay_bp.MinSumBPDecoderF64`). The min-sum scaling factor can be given as `ms_scaling_factor`
            for either backend (for "relay" it is passed on as `alpha`). For both backends, a value in (0, 1]
            scales the check-to-variable messages by that constant, and 0 uses the adaptive schedule
            1 - 2^-(iteration + 1).


        Returns
        -------
        BeliefMatching
            The BeliefMatching object for decoding using `model`
        """
        bm = cls.__new__(cls)
        model = circuit.detector_error_model(decompose_errors=True)
        bm._initialise_from_detector_error_model(
            model=model, max_bp_iters=max_bp_iters,
            bp_method=bp_method,
            bp_backend=bp_backend,
            **kwargs,
        )
        return bm

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode the syndrome and return a prediction of which observables were flipped

        Parameters
        ----------
        syndrome : np.ndarray
            A single shot of syndrome data. This should be a binary array with a length equal to the
            number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`. E.g. the syndrome might be
            one row of shot data sampled from a `stim.CompiledDetectorSampler`.

        Returns
        -------
        np.ndarray
            A binary numpy array `predictions` which predicts which observables were flipped.
            Its length is equal to the number of observables in the `stim.Circuit` or `stim.DetectorErrorModel`.
            `predictions[i]` is 1 if the decoder predicts observable `i` was flipped and 0 otherwise.
        """
        converged, corr, llrs = self._run_bp(syndrome)
        if converged:
            return (self._matrices.observables_matrix @ corr) % 2
        ps_h = 1 / (1 + np.exp(llrs))
        ps_e = self._matrices.hyperedge_to_edge_matrix @ ps_h
        eps = 1e-14
        ps_e[ps_e > 1 - eps] = 1 - eps
        ps_e[ps_e < eps] = eps
        weights = -np.log(ps_e)
        if self._edge_reweights is not None and weights.min() >= 0:
            # Fast path: temporarily reweight the prebuilt matching graph instead of rebuilding it.
            self._edge_reweights[:, 2] = weights
            return self._base_matching.decode(
                syndrome, edge_reweights=self._edge_reweights
            )
        matching = pymatching.Matching.from_check_matrix(
            self._matrices.edge_check_matrix,
            weights=weights,
            faults_matrix=self._matrices.edge_observables_matrix,
            use_virtual_boundary_node=True,
        )
        return matching.decode(syndrome)

    def decode_batch(self, shots: np.ndarray) -> np.ndarray:
        """
        Decode a batch of shots of syndrome data. This is just a helper method, equivalent to iterating over each
        shot and calling `BeliefMatching.decode` on it.

        Parameters
        ----------
        shots : np.ndarray
            A binary numpy array of dtype `np.uint8` or `bool` with shape `(num_shots, num_detectors)`, where
            here `num_shots` is the number of shots and `num_detectors` is the number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`.

        Returns
        -------
        np.ndarray
            A 2D numpy array `predictions` of dtype bool, where `predictions[i, :]` is the output of
            `self.decode(shots[i, :])`.
        """
        predictions = np.zeros(
            (shots.shape[0], self._matrices.observables_matrix.shape[0]), dtype=bool
        )
        for i in range(shots.shape[0]):
            predictions[i, :] = self.decode(shots[i, :])
        return predictions
