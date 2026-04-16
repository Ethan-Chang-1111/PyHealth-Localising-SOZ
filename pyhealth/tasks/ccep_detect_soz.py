"""PyHealth task for Seizure Onset Zone (SOZ) localisation from SPES responses.

Task paper:
    Norris et al. (ML4H 2024). "Localising the Seizure Onset Zone from
    Single-Pulse Electrical Stimulation Responses with a CNN Transformer."
    https://proceedings.mlr.press/v252/norris24a.html

Dataset:
    RESPectCCEPDataset — OpenNeuro ds004080
    van Blooijs et al. (2023). "CCEP ECoG dataset across age 4-51."
    https://openneuro.org/datasets/ds004080

Overview
--------
This task implements the **convergent** SPES analysis paradigm from Norris et al.
(2024): each candidate electrode ``e`` is classified as SOZ or non-SOZ from the
multi-channel matrix of mean CCEPs it *received* when every other electrode was
stimulated.  Critically, the dataset already handles all signal preprocessing
(bandpass filtering, epoching, baseline correction, resampling, and trial
averaging); the task's responsibility is to:

1. Collect all ``(recording_electrode=e, stim_site)`` rows from the patient's
   events.
2. Optionally filter stimulation sites that are too close to ``e`` using the
   per-row Euclidean distance derived from stored ``recording_x/y/z`` and the
   stimulation-electrode coordinates stored in a supplementary lookup.
3. Stack the per-channel mean response vectors into a ``[C × T]`` matrix.
4. Return one sample dict per labelled electrode.

Distance filtering note
-----------------------
The new dataset stores *recording* electrode coordinates (``recording_x/y/z``)
but does not store separate stimulation electrode coordinates.  When a patient
has coordinates for all electrodes, the distance between the recording electrode
``e`` and a stimulation pair ``(stim_1, stim_2)`` can be estimated as the
minimum distance to either pole of the pair, using a secondary lookup built
from *all* rows in which the electrode appears as a recording channel.  When
coordinates are unavailable (NaN), distance filtering is skipped and all
stimulation sites are included — a safe fallback consistent with datasets that
lack spatial metadata.

Class imbalance note
--------------------
Only ~14.4 % of electrodes carry a positive SOZ label (298 / 2066 in the full
cohort).  Downstream models should compensate, e.g. via
``torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.95]))``.

Usage
-----
>>> from pyhealth.datasets import RESPectCCEPDataset
>>> from pyhealth.datasets import split_by_patient, get_dataloader
>>> from pyhealth.tasks import SeizureOnsetZoneLocalisation
>>>
>>> dataset = RESPectCCEPDataset(root="/path/to/ds004080/")
>>> task = SeizureOnsetZoneLocalisation()
>>> samples = dataset.set_task(task)
>>> samples[0]
{
    'patient_id':    'sub-01',
    'visit_id':      'ses-1',
    'electrode_id':  'P22',
    'spes_responses': array([[...]], dtype=float32),  # shape [C, T]
    'stim_distances': array([...], dtype=float32),    # shape [C]
    'soz_label':     0,
}
>>> train_ds, val_ds, test_ds = split_by_patient(samples, [0.8, 0.1, 0.1])
>>> train_loader = get_dataloader(train_ds, batch_size=16, shuffle=True)
"""

import json
import logging
from collections import defaultdict
from typing import Any, cast, DefaultDict, Dict, List, Optional, Tuple

import numpy as np

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel used to signal "distance unknown, skip distance filter".
_DIST_UNKNOWN: float = -1.0

# Minimum inter-electrode distance (mm) to include a stimulation site as an
# input channel.  Sites closer than this risk volume-conduction artefacts from
# the stimulating electrode (Norris et al. 2024, §4.3; van Blooijs et al. 2023).
_DEFAULT_MIN_DISTANCE_MM: float = 13.0


# ---------------------------------------------------------------------------
# Public task class
# ---------------------------------------------------------------------------


class SeizureOnsetZoneLocalisation(BaseTask):
    """Localise the Seizure Onset Zone (SOZ) from SPES responses.

    Implements the **convergent** paradigm (Norris et al., ML4H 2024): each
    electrode is classified from the inward CCEPs it receives, not from the
    responses it evokes.  This approach yielded substantially higher AUROC
    (0.666 convergent vs. 0.574 divergent CNN baseline; 0.730 with the CNN
    Transformer that handles variable channel counts via cross-channel
    attention).

    Each ``Patient`` object exposes events of type ``"respectccep"`` with one
    event per ``(recording_electrode, stim_site)`` pair.  The task groups
    events by ``recording_electrode`` to build one convergent sample per
    labelled electrode.

    Parameters
    ----------
    min_distance_mm:
        Minimum Euclidean distance (mm) between the recording electrode and a
        stimulation site for that site's response to be included as an input
        channel.  Stimulation sites closer than this threshold are excluded to
        reduce volume-conduction artefacts.  Set to ``0.0`` to disable distance
        filtering entirely.  Default: 13.0 mm.

    Attributes
    ----------
    task_name : str
    input_schema : Dict[str, str]
        ``spes_responses`` — ``"timeseries"`` processor converts the
        ``[C, T]`` float32 array to a tensor.
        ``stim_distances`` — ``"timeseries"`` processor converts the
        ``[C]`` float32 array to a tensor.
    output_schema : Dict[str, str]
        ``soz_label`` — ``"binary"`` processor converts the ``{0, 1}`` int
        to a binary tensor.

    Sample dict keys
    ----------------
    patient_id : str
        BIDS subject identifier (e.g. ``"sub-01"``).
    visit_id : str
        BIDS session identifier (e.g. ``"ses-1"``), or ``"unknown"`` when the
        session is absent from the metadata.
    electrode_id : str
        Recording electrode label (e.g. ``"P22"``).
    spes_responses : np.ndarray, shape ``[C, T]``, dtype float32
        Stacked trial-averaged CCEP matrix.  ``C`` is the number of
        stimulation sites that passed the distance filter; ``T`` is the
        number of time points in each resampled epoch (determined by the
        dataset's ``tmin_s``, ``tmax_s``, and ``resample_hz`` parameters).
        **C varies across patients and electrodes** — models must handle
        variable-length channel axes (e.g. via cross-channel attention or
        random subsampling).
    stim_distances : np.ndarray, shape ``[C]``, dtype float32
        Euclidean distance (mm) from each stimulation site to the target
        recording electrode.  Rows correspond to the same stimulation sites as
        the rows of ``spes_responses``.  Entries are ``0.0`` when spatial
        coordinates were unavailable.  This vector is useful as an auxiliary
        feature for spatially-aware models (cf. the modified CNN Transformer
        with AUROC 0.745 reported in §4.7 of Norris et al. 2024).
    soz_label : int
        Binary SOZ membership label: ``1`` if the electrode is within the
        clinician-defined Seizure Onset Zone, ``0`` otherwise.
    """

    task_name: str = "SeizureOnsetZoneLocalisation"

    input_schema: Dict[str, str] = {
        "spes_responses": "timeseries",
        "stim_distances": "timeseries",
    }
    output_schema: Dict[str, str] = {
        "soz_label": "binary",
    }

    def __init__(self, min_distance_mm: float = _DEFAULT_MIN_DISTANCE_MM) -> None:
        super().__init__()
        if min_distance_mm < 0:
            raise ValueError(
                f"min_distance_mm must be >= 0, got {min_distance_mm}."
            )
        self.min_distance_mm = min_distance_mm

    # ------------------------------------------------------------------
    # Core BaseTask method
    # ------------------------------------------------------------------

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Generate SOZ-localisation samples for a single patient.

        Each call produces one sample dict per recording electrode that has
        (a) at least one valid inward SPES response after distance filtering
        and (b) a SOZ label (``0`` or ``1``).

        Parameters
        ----------
        patient:
            A ``pyhealth.data.Patient`` populated by ``RESPectCCEPDataset``.
            Each event of type ``"respectccep"`` must expose the attributes
            described in ``RESPectCCEPDataset._process_run``::

                event.recording_electrode   str
                event.stim_1                str   first electrode of stim pair
                event.stim_2                str   second electrode of stim pair
                event.response_ts           str   JSON-encoded float32 list
                event.soz_label             int   0 or 1
                event.session_id            str | None
                event.recording_x           float | nan
                event.recording_y           float | nan
                event.recording_z           float | nan

        Returns
        -------
        List[Dict[str, Any]]
            One sample dict per labelled electrode with the keys listed in the
            class docstring.  Returns an empty list when no usable data exist.
        """
        events: List[Event] = cast(
            List[Event],
            patient.get_events(event_type="respectccep", return_df=False),
        )
        if not events:
            logger.debug(
                "Patient %s: no 'respectccep' events found.",
                patient.patient_id,
            )
            return []

        # ----------------------------------------------------------------
        # Pass 1 — accumulate per-electrode metadata and per-(electrode,
        #           stim-site) response vectors from the pre-computed rows.
        # ----------------------------------------------------------------
        # electrode_meta[rec_id] = {"soz_label": int, "visit_id": str,
        #                            "coords": (x, y, z) | None}
        electrode_meta: Dict[str, Dict[str, Any]] = {}

        # stim_responses[rec_id][stim_site_key] = list of 1-D float32 arrays
        # stim_site_key is the canonical "E1-E2" label built from stim_1, stim_2.
        stim_responses: DefaultDict[str, DefaultDict[str, List[np.ndarray]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # coord_lookup[electrode_label] = (x, y, z) — built from recording-
        # side coordinates stored in events; used to estimate stim distances.
        coord_lookup: Dict[str, Tuple[float, float, float]] = {}
        logger.info(
            "Patient %s: Found %d events",
            patient.patient_id,
            len(events)
        )
        for event in events:
            rec_id = str(event.recording_electrode)
            stim_1 = str(event.stim_1)
            stim_2 = str(event.stim_2)
            stim_key = _canonical_stim_key(stim_1, stim_2)

            # ---- electrode-level metadata (stored on first encounter) --
            if rec_id not in electrode_meta:
                coords = _extract_coords(event)
                electrode_meta[rec_id] = {
                    "soz_label": int(getattr(event, "soz_label", 0)),
                    "visit_id": _safe_str(getattr(event, "session_id", None))
                    or "unknown",
                    "coords": coords,
                }
                if coords is not None:
                    coord_lookup[rec_id] = coords

            # ---- response timeseries -----------------------------------
            ts = getattr(event, "response_ts", None)
            if ts is None:
                continue
            arr = _parse_response_ts(ts)
            if arr is not None:
                stim_responses[rec_id][stim_key].append(arr)

        if not electrode_meta:
            return []

        # ----------------------------------------------------------------
        # Pass 2 — build one convergent sample per labelled electrode.
        # ----------------------------------------------------------------
        samples: List[Dict[str, Any]] = []
        test_rng = np.random.default_rng()
        for rec_id, meta in electrode_meta.items():
            soz_label: int = meta["soz_label"]
            # soz_label = test_rng.integers(0,2)
            logger.info("Sample has label %d", soz_label)

            rec_coords: Optional[Tuple[float, float, float]] = meta["coords"]
            if rec_id not in stim_responses:
                logger.debug(
                    "Patient %s electrode %s: no inward SPES responses.",
                    patient.patient_id,
                    rec_id,
                )
                continue

            per_site: DefaultDict[str, List[np.ndarray]] = stim_responses[rec_id]
            if not per_site:
                continue

            # ----------------------------------------------------------
            # For each stimulation site, compute distance to the recording
            # electrode and apply the min_distance_mm threshold.
            # ----------------------------------------------------------
            channel_responses: List[np.ndarray] = []
            channel_distances: List[float] = []

            for stim_key, response_list in per_site.items():
                if not response_list:
                    continue

                dist = _stim_distance(
                    stim_key=stim_key,
                    rec_coords=rec_coords,
                    coord_lookup=coord_lookup,
                )

                # When distance is unknown we include the channel
                # unconditionally (dist == _DIST_UNKNOWN).  When it is known
                # and below the threshold, exclude it.
                if dist != _DIST_UNKNOWN and dist < self.min_distance_mm:
                    continue

                # The dataset already averages across trials inside
                # _process_run; each list should normally contain exactly one
                # array.  We take the mean defensively in case multiple runs
                # contribute responses for the same stim site.
                mean_resp = _mean_responses(response_list)
                if mean_resp is None:
                    continue

                channel_responses.append(mean_resp)
                channel_distances.append(max(dist, 0.0))  # clamp _DIST_UNKNOWN → 0

            if not channel_responses:
                logger.debug(
                    "Patient %s electrode %s: no channels survived distance "
                    "filter (min_distance_mm=%.1f).",
                    patient.patient_id,
                    rec_id,
                    self.min_distance_mm,
                )
                continue

            # Stack → [C, T]
            response_matrix = np.stack(channel_responses, axis=0).astype(np.float32)
            distances_arr = np.array(channel_distances, dtype=np.float32)

            # TimeSeriesProcessor expects each timeseries field as a
            # (timestamps, values) 2-tuple.  timestamps is a 1-D index array
            # over the first axis of values (channels, in both cases here).
            n_channels = response_matrix.shape[0]
            channel_index = np.arange(n_channels, dtype=np.float32)
            
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": meta["visit_id"],
                    "electrode_id": rec_id,
                    "spes_responses": (channel_index, response_matrix),   # ([C], [C, T])
                    "stim_distances": (channel_index, distances_arr),     # ([C], [C])
                    "soz_label": soz_label,                               # int {0, 1}
                }
            )

        n_soz: int = sum(int(s["soz_label"]) for s in samples)
        logger.debug(
            "Patient %s: %d samples generated (%d SOZ, %d non-SOZ).",
            patient.patient_id,
            len(samples),
            n_soz,
            len(samples) - n_soz,
        )
        return samples


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _canonical_stim_key(stim_1: str, stim_2: str) -> str:
    """Return a canonical, order-independent key for a stimulation pair.

    The dataset's ``_canonical_stim_site`` already sorts the two poles, so
    this function mirrors that logic for consistency when looking up responses.

    Examples
    --------
    >>> _canonical_stim_key("P29", "P30")
    'P29-P30'
    >>> _canonical_stim_key("P30", "P29")
    'P29-P30'
    """
    parts = sorted([stim_1.strip(), stim_2.strip()])
    return f"{parts[0]}-{parts[1]}"


def _extract_coords(
    event: Any,
) -> Optional[Tuple[float, float, float]]:
    """Return ``(x, y, z)`` from an event's recording coordinate attributes.

    Returns ``None`` if any coordinate is missing or NaN.
    """
    try:
        x = float(event.recording_x)
        y = float(event.recording_y)
        z = float(event.recording_z)
    except (AttributeError, TypeError, ValueError):
        return None
    if any(np.isnan(v) for v in (x, y, z)):
        return None
    return (x, y, z)


def _safe_str(val: Any) -> Optional[str]:
    """Return ``str(val)`` unless ``val`` is ``None`` or float NaN."""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    return str(val)


def _parse_response_ts(ts: Any) -> Optional[np.ndarray]:
    """Decode a JSON-encoded or array-like response timeseries to float32.

    The dataset stores mean responses as compact JSON strings via
    ``RESPectCCEPDataset._to_json_1d``.  This function handles that format as
    well as plain list / array values for forward-compatibility.

    Returns ``None`` on any parse or shape error.
    """
    if ts is None:
        return None

    if isinstance(ts, str):
        try:
            data = json.loads(ts)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Could not parse response_ts JSON: %s", exc)
            return None
    else:
        data = ts

    try:
        arr = np.asarray(data, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        logger.warning("Could not convert response_ts to array: %s", exc)
        return None

    if arr.ndim != 1 or arr.size == 0:
        logger.warning(
            "response_ts has unexpected shape %s; expected 1-D non-empty array.",
            arr.shape,
        )
        return None

    return arr


def _stim_distance(
    stim_key: str,
    rec_coords: Optional[Tuple[float, float, float]],
    coord_lookup: Dict[str, Tuple[float, float, float]],
) -> float:
    """Estimate the distance (mm) between a stimulation pair and a recording electrode.

    Strategy
    --------
    The stimulation pair label ``stim_key`` has the form ``"E1-E2"``.  We look
    up coordinates for both poles from ``coord_lookup`` (which maps electrode
    labels to their ``(x, y, z)`` recording coordinates) and return the
    *minimum* distance to either pole.  This is conservative: if either pole is
    within ``min_distance_mm`` the site is excluded.

    Returns ``_DIST_UNKNOWN`` when either the recording or stimulation
    electrode coordinates are unavailable, signalling the caller to include the
    channel without filtering.

    Parameters
    ----------
    stim_key:
        Canonical stimulation pair label, e.g. ``"P29-P30"``.
    rec_coords:
        ``(x, y, z)`` of the recording electrode, or ``None``.
    coord_lookup:
        Mapping from electrode label to ``(x, y, z)`` built from all events.
    """
    if rec_coords is None:
        return _DIST_UNKNOWN

    poles = stim_key.split("-")
    distances: List[float] = []
    for pole in poles:
        pole_coords = coord_lookup.get(pole.strip())
        if pole_coords is None:
            # At least one pole has no coordinates — cannot filter.
            return _DIST_UNKNOWN
        distances.append(_euclidean_mm(rec_coords, pole_coords))

    return min(distances)


def _euclidean_mm(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    """Return Euclidean distance in mm between two 3-D coordinate tuples."""
    return float(np.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b))))


def _mean_responses(response_list: List[np.ndarray]) -> Optional[np.ndarray]:
    """Average a list of 1-D float32 response arrays.

    All arrays must share the same length.  Arrays with a mismatched length
    (e.g. from different resampling runs) are logged and skipped.  Returns
    ``None`` if no valid arrays remain.
    """
    if not response_list:
        return None

    # Use the first valid array's length as the reference.
    ref_len = response_list[0].size
    valid: List[np.ndarray] = []

    for arr in response_list:
        if arr.size != ref_len:
            logger.warning(
                "_mean_responses: length mismatch (%d vs expected %d); "
                "skipping array.",
                arr.size,
                ref_len,
            )
            continue
        valid.append(arr.astype(np.float32))

    if not valid:
        return None

    return np.mean(np.stack(valid, axis=0), axis=0).astype(np.float32)