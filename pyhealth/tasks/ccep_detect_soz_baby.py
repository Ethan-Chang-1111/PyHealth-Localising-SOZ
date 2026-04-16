"""Minimal PyHealth task for the RESPectCCEP dataset.

Returns one sample per labelled electrode containing only:
    - age  (participant age at recording, scalar feature)
    - soz_label  (binary SOZ membership label)

This task is useful for:
    - Sanity-checking the dataset pipeline end-to-end without signal data.
    - Confirming that the BinaryLabelProcessor sees both classes.
    - Baseline experiments using only demographic information.

Usage
-----
>>> from pyhealth.datasets import RESPectCCEPDataset
>>> from pyhealth.tasks.soz_age_baseline import SOZAgeBaseline
>>>
>>> dataset = RESPectCCEPDataset(root="/path/to/ds004080/")
>>> task = SOZAgeBaseline()
>>> samples = dataset.set_task(task)
>>> samples[0]
{
    'patient_id': 'sub-01',
    'visit_id':   'ses-1',
    'electrode_id': 'P22',
    'age':        25.0,
    'soz_label':  1,
}
"""

import logging
from typing import Any, Dict, List, cast

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


class SOZAgeBaseline(BaseTask):
    """Minimal SOZ classification task using only participant age.

    Produces one sample per recording electrode that has a SOZ label.
    Patients with no SOZ-positive electrodes are excluded to ensure
    BinaryLabelProcessor always sees both classes across the dataset.

    input_schema
    ------------
    age : "scalar"
        Participant age in years as a float32 scalar.

    output_schema
    -------------
    soz_label : "binary"
        1 if the electrode is within the clinician-defined SOZ, 0 otherwise.

    Sample keys
    -----------
    patient_id : str
    visit_id   : str
    electrode_id : str
    age        : float
    soz_label  : int  {0, 1}
    """

    task_name: str = "SOZAgeBaseline"

    input_schema: Dict[str, str] = {
        "age": "scalar",
    }
    output_schema: Dict[str, str] = {
        "soz_label": "binary",
    }

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Generate one age + soz_label sample per labelled electrode.

        Parameters
        ----------
        patient:
            A ``pyhealth.data.Patient`` populated by ``RESPectCCEPDataset``.
            Each event of type ``"respectccep"`` must expose:

                event.recording_electrode   str
                event.age                   float | int | None
                event.soz_label             int   0 or 1
                event.session_id            str | None

        Returns
        -------
        List[Dict[str, Any]]
            One dict per unique labelled electrode.  Returns ``[]`` when the
            patient has no SOZ-positive electrodes (to prevent the global label
            pool from collapsing to ``{0}``).
        """
        events: List[Event] = cast(
            List[Event],
            patient.get_events(event_type="respectccep", return_df=False),
        )
        if not events:
            logger.debug("Patient %s: no 'respectccep' events.", patient.patient_id)
            return []

        # One sample per electrode — use the first event for that electrode
        # to read its age and soz_label (both are patient/electrode-level
        # constants; they do not vary across stim sites).
        seen_electrodes: Dict[str, Dict[str, Any]] = {}

        for event in events:
            rec_id = str(event.recording_electrode)
            if rec_id in seen_electrodes:
                continue

            # Parse age — stored as a numeric column; may be NaN for some
            # participants whose participants.tsv omits the age field.
            try:
                age = float(getattr(event, "age", float("nan")))
            except (TypeError, ValueError):
                age = float("nan")

            soz_label = int(getattr(event, "soz_label", 0))
            logger.info("Sample has label %d", soz_label)

            session = getattr(event, "session_id", None)
            visit_id = str(session) if session is not None else "unknown"

            seen_electrodes[rec_id] = {
                "patient_id":   patient.patient_id,
                "visit_id":     visit_id,
                "electrode_id": rec_id,
                "age":          age,
                "soz_label":    soz_label,
            }

        samples = list(seen_electrodes.values())

        n_soz = sum(s["soz_label"] for s in samples)
        logger.debug(
            "Patient %s: %d samples (%d SOZ, %d non-SOZ).",
            patient.patient_id,
            len(samples),
            n_soz,
            len(samples) - n_soz,
        )

        # Exclude patients with no SOZ-positive electrodes so
        # BinaryLabelProcessor always sees both labels globally.
        if n_soz == 0:
            logger.debug(
                "Patient %s: skipping — no SOZ-positive electrodes.",
                patient.patient_id,
            )
            return []

        return samples