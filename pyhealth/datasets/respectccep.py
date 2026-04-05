"""PyHealth dataset implementation for RESPect CCEP (OpenNeuro ds004080).

Dataset link:
    https://openneuro.org/datasets/ds004080

This dataset class follows the common PyHealth pattern for file-based datasets:
it builds a lightweight metadata CSV index (one row per BIDS run) and stores
file pointers to raw signal/annotation sidecars rather than embedding raw
waveforms directly in tabular metadata.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class RESPectCCEPDataset(BaseDataset):
    """Base dataset for RESPect CCEP (OpenNeuro ds004080).

    This dataset indexes the BIDS iEEG directory into one metadata row per run
    (``*_events.tsv``), then lets ``BaseDataset`` expose those rows as PyHealth
    events.

    Indexing behavior:
    - Scans ``sub-*/ses-*/ieeg/*_events.tsv`` files.
    - Extracts BIDS entities: ``participant_id``, ``session_id``, ``run_id``.
    - Attaches participant demographics from ``participants.tsv``.
    - Resolves run/session sidecars (channels, electrodes, coordsystem, BrainVision triplet, JSON).
    - Writes ``respect_ccep_metadata-pyhealth.csv`` under ``root``.

    Notes:
    - Set `root` to the repository directory that includes the dataset files and folders.
    - Metadata rows keep relative file paths so downstream tasks can load
      only the raw files they need.
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the RESPect CCEP dataset.

        Args:
            root: Root directory of the BIDS dataset.
            dataset_name: Optional custom dataset name. Defaults to
                ``"respect_ccep"``.
            config_path: Optional path to dataset YAML config. If not provided,
                uses ``pyhealth/datasets/configs/respect_ccep.yaml``.
            **kwargs: Extra keyword arguments forwarded to ``BaseDataset``.

        Raises:
            FileNotFoundError: If required dataset files (e.g.,
                ``participants.tsv``) are missing during metadata creation.
            ValueError: If ``participants.tsv`` lacks required columns.
        """
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "respect_ccep.yaml"

        self._pyhealth_csv = str(Path(root) / "respect_ccep_metadata-pyhealth.csv")
        if not Path(self._pyhealth_csv).exists():
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=["respectccep"],
            dataset_name=dataset_name or "respect_ccep",
            config_path=config_path,
            **kwargs,
        )

    @staticmethod
    def _extract_bids_entities(file_path: Path) -> Dict[str, Optional[str]]:
        """Extract basic BIDS entities from a run filename.

        Args:
            file_path: Path to a BIDS-like file (typically ``*_events.tsv``).

        Returns:
            A dictionary containing ``participant_id``, ``session_id``, and
            ``run_id``. Missing entities are set to ``None``.
        """
        entities: Dict[str, Optional[str]] = {
            "participant_id": None,
            "session_id": None,
            "run_id": None,
        }
        for part in file_path.stem.split("_"):
            if part.startswith("sub-"):
                entities["participant_id"] = part
            elif part.startswith("ses-"):
                entities["session_id"] = part
            elif part.startswith("run-"):
                entities["run_id"] = part
        return entities

    @staticmethod
    def _relative_or_none(path: Optional[Path], root_path: Path) -> Optional[str]:
        """Convert an absolute path to a root-relative POSIX path.

        Args:
            path: Candidate path.
            root_path: Dataset root used as the relativization base.

        Returns:
            Relative POSIX path string if ``path`` exists, else ``None``.
        """
        if path is None or not path.exists():
            return None
        return path.relative_to(root_path).as_posix()

    @staticmethod
    def _find_session_file(
        ieeg_dir: Path,
        participant_id: Optional[str],
        session_id: Optional[str],
        suffix: str,
    ) -> Optional[Path]:
        """Locate a session-level BIDS sidecar in an ``ieeg`` directory.

        Resolution strategy:
        1. ``sub-*_ses-*_{{suffix}}``
        2. ``sub-*_*{{suffix}}``
        3. first glob match for ``*_{suffix}``

        Args:
            ieeg_dir: Directory containing iEEG files for a session.
            participant_id: BIDS participant entity (e.g., ``sub-01``).
            session_id: BIDS session entity (e.g., ``ses-1``).
            suffix: Target filename suffix (e.g., ``"electrodes.tsv"``).

        Returns:
            First matching path if found, otherwise ``None``.
        """
        candidates: List[Path] = []
        if participant_id and session_id:
            candidates.append(ieeg_dir / f"{participant_id}_{session_id}_{suffix}")
        if participant_id:
            candidates.append(ieeg_dir / f"{participant_id}_{suffix}")
        candidates.extend(sorted(ieeg_dir.glob(f"*_{suffix}")))
        for candidate in candidates:
            if candidate.exists() and ":Zone.Identifier" not in candidate.name:
                return candidate
        return None

    @staticmethod
    def _find_run_sidecar(events_path: Path, suffix: str) -> Optional[Path]:
        """Locate a run-level sidecar based on an ``*_events.tsv`` file.

        Args:
            events_path: Path to the run events file.
            suffix: Desired companion suffix (e.g., ``"_ieeg.vhdr"``).

        Returns:
            Matching sidecar path if present, else ``None``.
        """
        expected = events_path.with_name(events_path.name.replace("_events.tsv", suffix))
        if expected.exists():
            return expected
        return None

    def prepare_metadata(self, root: str) -> None:
        """Create ``respect_ccep_metadata-pyhealth.csv`` from BIDS folders.

        The output CSV contains one row per run and is designed to be consumed
        by ``BaseDataset`` via ``respect_ccep.yaml``.

        Args:
            root: BIDS dataset root directory.

        Raises:
            FileNotFoundError: If the root or ``participants.tsv`` is missing.
            ValueError: If ``participants.tsv`` has no ``participant_id``.
        """
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        participants_path = root_path / "participants.tsv"
        if not participants_path.exists():
            raise FileNotFoundError(
                f"participants.tsv not found in dataset root: {participants_path}"
            )

        participants_df = pd.read_csv(participants_path, sep="\t")
        if "participant_id" not in participants_df.columns:
            raise ValueError("participants.tsv must contain a 'participant_id' column")

        participant_lookup: Dict[str, Dict[str, object]] = (
            participants_df.drop_duplicates(subset=["participant_id"], keep="first")
            .set_index("participant_id")
            .to_dict(orient="index")
        )
        participant_session_lookup: Dict[tuple[str, str], Dict[str, object]] = {}
        if "session" in participants_df.columns:
            participant_session_lookup = (
                participants_df.drop_duplicates(
                    subset=["participant_id", "session"], keep="first"
                )
                .set_index(["participant_id", "session"])
                .to_dict(orient="index")
            )

        rows: List[Dict[str, object]] = []
        event_files: List[Path] = []
        for subject_dir in sorted(root_path.glob("sub-*")):
            if not subject_dir.is_dir():
                continue
            event_files.extend(sorted(subject_dir.glob("ses-*/ieeg/*_events.tsv")))

        for events_path in event_files:
            if ":Zone.Identifier" in events_path.name:
                continue

            entities = self._extract_bids_entities(events_path)
            participant_id = entities["participant_id"]
            session_id = entities["session_id"]
            run_id = entities["run_id"]

            if participant_id is None:
                logger.warning("Skipping malformed BIDS file without subject: %s", events_path)
                continue

            ieeg_dir = events_path.parent
            channels_path = self._find_run_sidecar(events_path, "_channels.tsv")
            vhdr_path = self._find_run_sidecar(events_path, "_ieeg.vhdr")
            eeg_path = self._find_run_sidecar(events_path, "_ieeg.eeg")
            vmrk_path = self._find_run_sidecar(events_path, "_ieeg.vmrk")
            ieeg_json_path = self._find_run_sidecar(events_path, "_ieeg.json")
            electrodes_path = self._find_session_file(
                ieeg_dir, participant_id, session_id, "electrodes.tsv"
            )
            coordsystem_path = self._find_session_file(
                ieeg_dir, participant_id, session_id, "coordsystem.json"
            )

            demographics = participant_lookup.get(participant_id, {}).copy()
            if session_id is not None:
                session_demographics = participant_session_lookup.get(
                    (participant_id, session_id), {}
                )
                demographics.update(session_demographics)

            rows.append(
                {
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "run_id": run_id,
                    "age": demographics.get("age"),
                    "sex": demographics.get("sex"),
                    "participant_session": demographics.get("session"),
                    "events_file_path": self._relative_or_none(events_path, root_path),
                    "channels_file_path": self._relative_or_none(channels_path, root_path),
                    "electrodes_file_path": self._relative_or_none(electrodes_path, root_path),
                    "coordsystem_file_path": self._relative_or_none(coordsystem_path, root_path),
                    "vhdr_file_path": self._relative_or_none(vhdr_path, root_path),
                    "vmrk_file_path": self._relative_or_none(vmrk_path, root_path),
                    "eeg_file_path": self._relative_or_none(eeg_path, root_path),
                    "ieeg_json_file_path": self._relative_or_none(ieeg_json_path, root_path),
                }
            )

        metadata_cols = [
            "participant_id",
            "session_id",
            "run_id",
            "age",
            "sex",
            "participant_session",
            "events_file_path",
            "channels_file_path",
            "electrodes_file_path",
            "coordsystem_file_path",
            "vhdr_file_path",
            "vmrk_file_path",
            "eeg_file_path",
            "ieeg_json_file_path",
        ]
        metadata_df = pd.DataFrame(rows, columns=metadata_cols)
        metadata_df.to_csv(self._pyhealth_csv, index=False)
        logger.info("Wrote %d RESPect CCEP run records to %s", len(metadata_df), self._pyhealth_csv)
