import os
import traceback
import uuid
from time import time
from typing import List

import torch


class PredictTask:
    def __init__(self, filenames: str | List[str], ofile: str, id: uuid.UUID = None):
        filenames = [filenames] if isinstance(filenames, str) else list(filenames)
        fn0 = filenames[0]
        name = os.path.splitext(os.path.basename(fn0))[0].rsplit('_', 1)[0]
        self.id = uuid.uuid4() if id is None else id
        self.name: str = name
        self.filenames: List[str] = filenames

        if not os.path.splitext(ofile)[1]:
            # no extension means ofile is a directory
            ofile = os.path.join(ofile, f"{name}.nrrd")
        self.ofile: str = ofile

        self.save_probabilities: bool = False
        self.overwrite: bool = True

        self.success = False
        self.error = None
        self.done = False

        self.timestamps = dict()
        self.timestamps['start'] = time()

    def print(self):
        status = ("failed" if self.error is not None
                  else ("finished" if self.success
                        else ("aborted" if self.done else "pending")))


        times = "".join([f"\n\t * {k}: {v - self.timestamps['start']:.02f}"
                         for k, v in self.timestamps.items() if k != 'start'])

        print(f"Task {self.id}: {status}\n"
              f"- input: {self.name}\n"
              f"- output: {self.ofile}\n"
              f"- times: {times}")

def _run_worker(predictor, task_queue, done, results):
    try:
        print(f"A new worker started running ({os.getpid()})")
        while True:
            try:
                task = task_queue.get()
            except EOFError:
                # the queue was closed...
                return

            if task is None:
                # soft shutdown communicated through the queue
                break
            try:
                task.timestamps['get'] = time()
                _run_predict(predictor, task)
                task.success = True
            except Exception as e:
                task.error = e
            finally:
                if task is not None:
                    task.timestamps['done'] = time()
                    task.done = True
                    with done:
                        results[task.id] = task
                        done.notify()
    except (KeyboardInterrupt, SystemExit) as ex:
        print(f"Worker {os.getpid()} was killed: ({type(ex).__name__})")
    except Exception as ex:
        print(f"Worker {os.getpid()} crashed: ({type(ex).__name__})")
        traceback.print_exc()
    finally:
        #print(f"Worker concluded ({os.getpid()})")
        pass

def _run_predict(predictor, task: PredictTask):
    try:
        ofile_truncated, ofile_ext = os.path.splitext(task.ofile)
        expected_ext = predictor.dataset_json['file_ending']
        if expected_ext != ofile_ext:
            raise RuntimeError(f"Expected '{expected_ext}' file extension for ofile, found: {ofile_ext}")
    except Exception as ex:
        raise RuntimeError(f"Failed parsing the output file name: {ex}")

    try:
        odir = os.path.dirname(task.ofile)
        if not os.path.exists(odir):
            os.makedirs(odir, exist_ok=True)
    except Exception as ex:
        raise RuntimeError(f"Could not create output directory: {ex}")

    try:
        preprocessor = predictor.configuration_manager.preprocessor_class(verbose=predictor.verbose)
        data, _, data_properties = preprocessor.run_case(task.filenames,
                                                         None,
                                                         predictor.plans_manager,
                                                         predictor.configuration_manager,
                                                         predictor.dataset_json)
        task.timestamps['preprocessed'] = time()
    except Exception as ex:
        raise RuntimeError(f"Preprocessing failed for {task.name}: {ex}")

    try:
        data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)
        if predictor.device.type == 'cuda':
            data.pin_memory()
        prediction = predictor.predict_logits_from_preprocessed_data(data).cpu()
        task.timestamps['predicted'] = time()
    except Exception as ex:
        raise RuntimeError(f"Prediction failed for {task.name}: {ex}")

    try:
        from nnunetv2.inference.export_prediction import export_prediction_from_logits
        export_prediction_from_logits(prediction, data_properties,
                                      predictor.configuration_manager,
                                      predictor.plans_manager,
                                      predictor.dataset_json,
                                      ofile_truncated,
                                      task.save_probabilities)
        task.timestamps['exported'] = time()
    except Exception as ex:
        raise RuntimeError(f"Export failed for {task.name}: {ex}")
