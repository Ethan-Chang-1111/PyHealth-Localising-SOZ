# """
# PyHealth task for Seizure Onset Zone using the RESPectCCEP Dataset

# Dataset link:
#     https://openneuro.org/datasets/ds004080/versions/1.2.4

# Dataset paper: (please cite if you use this dataset)
#     D. van Blooijs, M.A. van den Boom, J.F. van der Aar, G.J.M. Huiskamp, G. Castegnaro,
#     M. Demuru, W.J.E.M. Zweiphenning, P. van Eijsden, K. J. Miller, F.S.S. Leijten, and
#     D. Hermes. ”ccep ecog dataset across age 4-51”, 2023a.

# Dataset paper link:
#     https://doi.org/10.1038/s41593-023-01272-0

# Author:
#     Ethan Chang (ethanc8@illinois.edu)
# """

# import logging
# from typing import Dict, List

# from pyhealth.data import Event, Patient
# from pyhealth.tasks import BaseTask

# from pyhealth.datasets import ChestXray14Dataset

# logger = logging.getLogger(__name__)


# class SeizureOnsetZoneLocalisation(BaseTask):
#     """
#     A PyHealth task class for binary classification of a specific disease
#     in the RESPectCCEP dataset.

#     Attributes:
#         task_name (str): The name of the task.
#         input_schema (Dict[str, str]): The schema for the task input.
#         output_schema (Dict[str, str]): The schema for the task output.

#     Examples:
#         >>> from pyhealth.datasets import RESPectCCEPDataset
#         >>> from pyhealth.tasks import SeizureOnsetDetection
        
#         >>> dataset = RESPectCCEPDataset(root="/path/to/ds004080/")
#         >>> task = SeizureOnsetDetection()
#         >>> samples = dataset.set_task(task)
#     """

#     task_name: str = "SeizureOnsetDetection"
#     input_schema: Dict[str, str] = {"image": "image"}
#     output_schema: Dict[str, str] = {"label": "binary"}

#     def __init__() -> None:
#         """
#         Initializes the SeizureOnsetDetection task

#         Args:

#         Raises:
#             ValueError: If the specified disease is not a valid class in the dataset.
#         """
        

#     def __call__(self, patient: Patient) -> List[Dict]:
#         """
#         Generates binary classification data samples for a single patient.

#         Args:
#             patient (Patient): A patient object containing at least one
#                                'respectccep' event.

#         Returns:
#             List[Dict]: A list containing a dictionary for each patient visit with:
#                 - 'image': path to the chest X-ray image.
#                 - 'label': binary label for the specified disease.
#         """
#         events: List[Event] = patient.get_events(event_type="chestxray14")

#         samples = []
#         for event in events:
#             samples.append({"image": event["path"], "label": int(event[self.disease])})

#         return samples
