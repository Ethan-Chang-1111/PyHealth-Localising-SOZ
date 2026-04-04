"""PyHealth task for Seizure Onset Zone (SOZ) localisation from SPES responses.

Task paper:
    Norris et al. (ML4H 2024). "Localising the Seizure Onset Zone from
    Single-Pulse Electrical Stimulation Responses with a CNN Transformer."
    https://proceedings.mlr.press/v252/norris24a.html

Dataset:
    RESPectCCEPDataset — OpenNeuro ds004080
    van Blooijs et al. (2023). "CCEP ECoG dataset across age 4-51."
    https://openneuro.org/datasets/ds004080

This module implements the convergent SPES paradigm described in Norris et al.
(2024): each candidate electrode is classified as SOZ / non-SOZ based on the
*inward* cortico-cortical evoked potentials (CCEPs) it receives when all other
electrodes are stimulated.  This contrasts with the more common divergent
paradigm (classify the *stimulating* electrode from the responses it evokes),
and was shown by the paper to yield substantially higher AUROC (0.666 vs 0.574
for CNNconvergent vs CNNdivergent, rising to 0.730 with the CNN Transformer).

Preprocessing applied here mirrors the paper (Section 4.3):
    - Stimulation artefact window (0–9 ms post-stimulus) is excluded.
    - Electrodes within 13 mm of the stimulated electrode are excluded from
      the response channels for that trial.
    - Trial responses are averaged across repetitions to improve SNR
      (this attenuates delayed responses but is the approach used in the paper).
    - Each per-channel, per-stimulation-site average forms one row of the
      multi-channel input matrix X ∈ R^{C × T} for a target electrode.

Class imbalance note:
    Only ~14.4% of electrodes are labelled SOZ (298 / 2066 in the full cohort).
    Downstream models should use weighted loss, e.g. BCEWithLogitsLoss with
    pos_weight = (1 - soz_rate) / soz_rate ≈ 5.95.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pyhealth.data import Patient
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants matching the paper's preprocessing (Norris et al., 2024, §4.3)
# ---------------------------------------------------------------------------

#: Sampling rate of the ECoG recordings (Hz).
SAMPLING_RATE_HZ: int = 2048

#: Width of the stimulation artefact blanking window (seconds).
ARTEFACT_BLANK_S: float = 0.009  # 9 ms

#: End of the analysis window (seconds post-stimulus).
EPOCH_END_S: float = 1.0  # 1 s

#: Minimum inter-electrode distance (mm) to retain as a recording channel
#: for a given stimulation site.  Channels closer than this are excluded to
#: avoid volume-conduction artefacts from the stimulating electrode.
MIN_DISTANCE_MM: float = 13.0

# Derived sample counts (artefact-blanked epoch length)
_ARTEFACT_SAMPLES: int = int(ARTEFACT_BLANK_S * SAMPLING_RATE_HZ)  # ≈ 18
_EPOCH_SAMPLES: int = int(EPOCH_END_S * SAMPLING_RATE_HZ)  # 2048
_RESPONSE_SAMPLES: int = _EPOCH_SAMPLES - _ARTEFACT_SAMPLES  # ≈ 2030


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------


class SeizureOnsetZoneLocalisation(BaseTask):
    """Localise the Seizure Onset Zone (SOZ) from SPES responses.

    Implements the **convergent** paradigm from Norris et al. (ML4H 2024):
    each electrode `e` is classified as SOZ / non-SOZ from the multi-channel
    matrix of trial-averaged CCEPs *received* at `e` when every other
    electrode (≥ ``min_distance_mm`` away) was stimulated in turn.

    Each call to this task on a single ``Patient`` yields one sample per
    electrode that (a) appears as a recording site and (b) has a SOZ label.
    Electrodes without SOZ annotations are silently skipped so the task
    degrades gracefully on the 39 patients in ds004080 who lack SOZ labels.

    Task schema
    -----------
    Input
        ``spes_responses`` : array of shape ``[C, T]``
            Trial-averaged CCEP matrix.  ``C`` is the number of stimulation
            sites that were ≥ ``min_distance_mm`` from electrode `e` (varies
            per patient and electrode), ``T = _RESPONSE_SAMPLES ≈ 2030``.
        ``stim_distances`` : array of shape ``[C]``
            Euclidean distance (mm) between each stimulation electrode and
            the target electrode `e`.  Useful as an auxiliary feature for
            models that incorporate spatial context (cf. the modified CNN
            Transformer with AUROC 0.745 in §4.7 of the paper).
    Output
        ``soz_label`` : binary int (0 = non-SOZ, 1 = SOZ)

    Parameters
    ----------
    min_distance_mm:
        Minimum distance (mm) between a stimulation electrode and the target
        recording electrode for the stimulation trial to be included.
        Default: 13 mm (matching the paper's preprocessing).
    require_soz_label:
        If ``True`` (default), silently drop electrodes that have no SOZ
        annotation.  Set to ``False`` to raise ``ValueError`` instead, which
        is useful during unit testing with fully-annotated synthetic data.

    Examples
    --------
    >>> from pyhealth.datasets import RESPectCCEPDataset
    >>> from pyhealth.datasets import split_by_patient, get_dataloader
    >>> from pyhealth.tasks import SeizureOnsetZoneLocalisation
    >>>
    >>> dataset = RESPectCCEPDataset(root="/path/to/ds004080/")
    >>> task = SeizureOnsetZoneLocalisation()
    >>> samples = dataset.set_task(task)
    >>> samples[0]
    {
        'patient_id': 'sub-01',
        'visit_id':   'ses-1',
        'electrode_id': 'P22',
        'spes_responses': array([[...]], dtype=float32),  # shape [C, T]
        'stim_distances': array([...], dtype=float32),    # shape [C]
        'soz_label': 0,
    }
    >>>
    >>> train_ds, val_ds, test_ds = split_by_patient(samples, [0.8, 0.1, 0.1])
    >>> train_loader = get_dataloader(train_ds, batch_size=16, shuffle=True)
    """

    # ------------------------------------------------------------------
    # BaseTask required class attributes
    # ------------------------------------------------------------------

    task_name: str = "SeizureOnsetZoneLocalisation"

    # ``timeseries`` processor converts the numpy array to a float32 tensor;
    # ``binary`` processor converts the int label to a {0,1} tensor.
    input_schema: Dict[str, str] = {
        "spes_responses": "timeseries",
        "stim_distances": "timeseries",
    }
    output_schema: Dict[str, str] = {
        "soz_label": "binary",
    }

    def __init__(
        self,
        min_distance_mm: float = MIN_DISTANCE_MM,
        require_soz_label: bool = True,
    ) -> None:
        super().__init__()
        self.min_distance_mm = min_distance_mm
        self.require_soz_label = require_soz_label

    # ------------------------------------------------------------------
    # BaseTask required method
    # ------------------------------------------------------------------

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Generate SOZ-localisation samples for one patient.

        Parameters
        ----------
        patient:
            A ``pyhealth.data.Patient`` object populated by
            ``RESPectCCEPDataset``.  The patient must expose events of type
            ``"respectccep"`` with at least the following attributes:

            - ``electrode_id`` (str): BIDS electrode label (e.g. ``"P22"``).
            - ``stim_electrode_id`` (str): electrode that was stimulated.
            - ``response_timeseries`` (array-like, length ≥ ``_EPOCH_SAMPLES``):
              raw epoch from 0 to 1 s post-stimulus at ``SAMPLING_RATE_HZ``.
            - ``soz_label`` (int | float | None): 1 if in SOZ, 0 if not.
            - ``x_coord``, ``y_coord``, ``z_coord`` (float | None):
              MNI or native-space coordinates in mm used to compute
              inter-electrode distances.

        Returns
        -------
        List[Dict[str, Any]]
            One dict per target electrode with the keys described in the class
            docstring.  Returns an empty list if the patient has no usable
            labelled electrodes or no SPES events.
        """
        samples: List[Dict[str, Any]] = []

        # ----------------------------------------------------------------
        # 1. Pull all SPES events for this patient.
        # ----------------------------------------------------------------
        events = patient.get_events(event_type="respectccep")
        if not events:
            logger.debug(
                "Patient %s has no 'respectccep' events — skipping.",
                patient.patient_id,
            )
            return samples

        # ----------------------------------------------------------------
        # 2. Build per-electrode metadata and coordinate lookup tables.
        # ----------------------------------------------------------------
        # electrode_meta: electrode_id -> dict with 'soz_label', 'coords'
        # spes_trials:   record_electrode_id ->
        #                    {stim_electrode_id -> list of response arrays}
        electrode_meta: Dict[str, Dict[str, Any]] = {}
        spes_trials: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(
            lambda: defaultdict(list)
        )
        stim_coords: Dict[str, Optional[Tuple[float, float, float]]] = {}

        for event in events:
            rec_id: str = str(event.electrode_id)
            stim_id: str = str(event.stim_electrode_id)
            soz_raw = getattr(event, "soz_label", None)
            coords = _safe_coords(event)

            # Register electrode metadata on first encounter.
            if rec_id not in electrode_meta:
                soz_label = _parse_soz_label(soz_raw)
                electrode_meta[rec_id] = {
                    "soz_label": soz_label,
                    "coords": coords,
                    "visit_id": _safe_str(getattr(event, "session_id", None))
                    or "unknown",
                }

            # Register stimulation electrode coordinates.
            if stim_id not in stim_coords:
                stim_coords[stim_id] = coords  # proxy: first-seen coords

            # Store raw response timeseries for (record, stim) pair.
            raw = getattr(event, "response_timeseries", None)
            if raw is not None:
                try:
                    arr = np.asarray(raw, dtype=np.float32)
                    spes_trials[rec_id][stim_id].append(arr)
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Patient %s: could not convert response for "
                        "rec=%s stim=%s — %s",
                        patient.patient_id,
                        rec_id,
                        stim_id,
                        exc,
                    )

        if not electrode_meta:
            return samples

        # ----------------------------------------------------------------
        # 3. Pre-compute a distance matrix between all electrode pairs.
        # ----------------------------------------------------------------
        distance_mm: Dict[str, Dict[str, float]] = _build_distance_matrix(
            {eid: meta["coords"] for eid, meta in electrode_meta.items()},
            stim_coords,
        )

        # ----------------------------------------------------------------
        # 4. Build one convergent sample per labelled recording electrode.
        # ----------------------------------------------------------------
        for rec_id, meta in electrode_meta.items():
            soz_label = meta["soz_label"]

            # Skip unlabelled electrodes.
            if soz_label is None:
                if not self.require_soz_label:
                    raise ValueError(
                        f"Patient {patient.patient_id}: electrode '{rec_id}' "
                        "has no SOZ label but require_soz_label=False "
                        "was set — check your data."
                    )
                logger.debug(
                    "Patient %s electrode %s has no SOZ label — skipping.",
                    patient.patient_id,
                    rec_id,
                )
                continue

            stim_pairs = spes_trials.get(rec_id, {})
            if not stim_pairs:
                logger.debug(
                    "Patient %s electrode %s has no inward SPES trials — skipping.",
                    patient.patient_id,
                    rec_id,
                )
                continue

            # ----------------------------------------------------------
            # 4a. Filter stimulation sites by distance and average trials.
            # ----------------------------------------------------------
            channel_responses: List[np.ndarray] = []
            channel_distances: List[float] = []

            for stim_id, trial_list in stim_pairs.items():
                dist = distance_mm.get(rec_id, {}).get(stim_id, 0.0)
                if dist < self.min_distance_mm:
                    # Exclude near-field channels (volume conduction artefact).
                    continue

                avg_response = _average_trials(trial_list)
                if avg_response is None:
                    continue

                channel_responses.append(avg_response)
                channel_distances.append(dist)

            if not channel_responses:
                logger.debug(
                    "Patient %s electrode %s: no valid distant stim channels "
                    "after distance filtering — skipping.",
                    patient.patient_id,
                    rec_id,
                )
                continue

            # ----------------------------------------------------------
            # 4b. Stack channels → [C, T] input matrix.
            # ----------------------------------------------------------
            response_matrix = np.stack(channel_responses, axis=0)  # [C, T]
            distances_arr = np.array(channel_distances, dtype=np.float32)  # [C]

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": meta["visit_id"],
                    "electrode_id": rec_id,
                    "spes_responses": response_matrix,   # np.ndarray [C, T]
                    "stim_distances": distances_arr,     # np.ndarray [C]
                    "soz_label": int(soz_label),         # 0 or 1
                }
            )

        logger.debug(
            "Patient %s: generated %d SOZ-localisation samples "
            "(%d SOZ, %d non-SOZ).",
            patient.patient_id,
            len(samples),
            sum(s["soz_label"] for s in samples),
            sum(1 - s["soz_label"] for s in samples),
        )
        return samples


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def _parse_soz_label(raw: Any) -> Optional[int]:
    """Coerce a raw SOZ label attribute to ``{0, 1}`` or ``None``.

    Handles integer, float, string (``"SOZ"``/``"non-SOZ"``/``"1"``/``"0"``),
    and ``NaN`` / ``None`` gracefully.
    """
    if raw is None:
        return None
    if isinstance(raw, float) and np.isnan(raw):
        return None
    if isinstance(raw, (int, np.integer)):
        return int(bool(raw))
    if isinstance(raw, float):
        return int(bool(raw))
    if isinstance(raw, str):
        normed = raw.strip().lower()
        if normed in ("soz", "1", "true", "yes"):
            return 1
        if normed in ("non-soz", "nonsoz", "0", "false", "no"):
            return 0
        logger.warning("Unrecognised SOZ label string '%s' — treating as None.", raw)
        return None
    return None


def _safe_coords(
    event: Any,
) -> Optional[Tuple[float, float, float]]:
    """Extract (x, y, z) coordinates from an event object, or return None."""
    try:
        x = float(event.x_coord)
        y = float(event.y_coord)
        z = float(event.z_coord)
        if any(np.isnan(v) for v in (x, y, z)):
            return None
        return (x, y, z)
    except (AttributeError, TypeError, ValueError):
        return None


def _safe_str(val: Any) -> Optional[str]:
    """Return ``str(val)`` unless ``val`` is None or NaN-like."""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    return str(val)


def _euclidean_mm(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    """Return the Euclidean distance in mm between two 3-D coordinate tuples."""
    return float(np.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b))))


def _build_distance_matrix(
    rec_coords: Dict[str, Optional[Tuple[float, float, float]]],
    stim_coords: Dict[str, Optional[Tuple[float, float, float]]],
) -> Dict[str, Dict[str, float]]:
    """Return distance_mm[rec_id][stim_id] for all electrode pairs.

    If either coordinate is unavailable the distance defaults to 0.0, which
    means the channel will be excluded by the ``min_distance_mm`` filter —
    a conservative choice that avoids injecting near-field artefacts.
    """
    result: Dict[str, Dict[str, float]] = defaultdict(dict)
    for rec_id, r_coords in rec_coords.items():
        for stim_id, s_coords in stim_coords.items():
            if rec_id == stim_id:
                result[rec_id][stim_id] = 0.0
                continue
            if r_coords is None or s_coords is None:
                # Missing coordinates: default to 0 so the channel is excluded.
                result[rec_id][stim_id] = 0.0
            else:
                result[rec_id][stim_id] = _euclidean_mm(r_coords, s_coords)
    return result


def _average_trials(trial_list: List[np.ndarray]) -> Optional[np.ndarray]:
    """Average a list of raw trial arrays and return the artefact-blanked epoch.

    Each trial is expected to be at least ``_EPOCH_SAMPLES`` long.  The first
    ``_ARTEFACT_SAMPLES`` samples (0–9 ms) are discarded, leaving a
    ``[_RESPONSE_SAMPLES]``-length float32 vector.

    Returns ``None`` if the list is empty or all trials have invalid shape.
    """
    valid: List[np.ndarray] = []
    for arr in trial_list:
        if arr.ndim != 1:
            logger.warning(
                "_average_trials: unexpected array ndim=%d, skipping trial.",
                arr.ndim,
            )
            continue
        if len(arr) < _EPOCH_SAMPLES:
            # Truncated epoch — pad with zeros rather than discard entirely.
            pad = np.zeros(_EPOCH_SAMPLES, dtype=np.float32)
            pad[: len(arr)] = arr
            arr = pad
        valid.append(arr[:_EPOCH_SAMPLES].astype(np.float32))

    if not valid:
        return None

    averaged = np.mean(np.stack(valid, axis=0), axis=0)  # [_EPOCH_SAMPLES]
    return averaged[_ARTEFACT_SAMPLES:]  # blank artefact window → [_RESPONSE_SAMPLES]