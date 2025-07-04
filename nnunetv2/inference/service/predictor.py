import sys
import uuid
from multiprocessing.pool import AsyncResult
from time import time, sleep
from typing import List

import torch.multiprocessing as mp

from nnunetv2.inference.service.prediction_worker import _run_worker, PredictTask


class ParallelPredictor:
    def __init__(self, predictor, num_workers: int = 4, daemon=True, pool=True):
        self._workers: List[mp.Process] = []
        self._task_queue: mp.Queue = None
        self._task_lock: mp.Lock = None
        self._task_done: mp.Condition = None
        self._task_results: dict = None
        self._num_workers = num_workers
        self._predictor = predictor
        self._daemon = daemon
        self._pool = pool

    def start(self):
        manager = mp.Manager()

        self._task_lock = manager.Lock()
        self._task_queue = manager.Queue()
        self._task_done = manager.Condition()
        self._task_results = manager.dict()

        if  self._pool:
            # pools are faster to initialize than individual processes
            pool = mp.Pool(processes=self._num_workers)
            self._workers = [pool.apply_async(_run_worker,
                                                   args=[self._predictor, self._task_queue, self._task_done, self._task_results])
                             for _ in range(self._num_workers)]
        else:
            for rank in range(self._num_workers):
                p = mp.Process(target=_run_worker, args=[self._predictor, self._task_queue, self._task_done, self._task_results],
                               name=f"PredictionWorker#{rank}",
                               daemon=self._daemon)
                p.start()
                self._workers.append(p)

    def predict(self,
                filenames: str | List[str],
                ofile: str,
                id: uuid.UUID = None,
                save_probabilities: bool = False,
                overwrite: bool = True,
                wait=True):
        task = PredictTask(filenames, ofile, id=id)
        task.save_probabilities = save_probabilities
        task.overwrite = overwrite

        if self._task_queue is None:
            self.start()
        self._task_queue.put(task)

        if wait:
            return self.wait(task.id)
        return task.id

    def wait(self, ids: uuid.UUID | PredictTask | List[uuid.UUID] | List[PredictTask], timeout=None):
        multiple = not isinstance(ids, (uuid.UUID, PredictTask))
        wait_ids, results = (ids, dict()) if multiple else ([ids], None)
        wait_ids = set(t.id if isinstance(t, PredictTask) else t for t in wait_ids)
        start = time()
        while True:
            with self._task_done:
                self._task_done.wait()
                for id in list(wait_ids):
                    result = self._task_results.pop(id, None)
                    if result is not None:
                        if multiple:
                            wait_ids.remove(id)
                            results[id] = result
                        else:
                            return result
            if not wait_ids:
                break
            self._stop_wait(0)
            with self._task_lock:
                if not self._workers:
                    # all workers have ended
                    break
            sleep(0.01)

            if timeout is not None and time() - start < timeout:
                return results
        return results

    def stop(self, timeout=10):
        # soft shutdown
        for _ in range(self._num_workers):
            self._task_queue.put(None)
        if not self._stop_wait(timeout):
            print("Soft shutdown did not work, terminating workers...", file=sys.stderr)
            # terminate processes
            with self._task_lock:
                workers = list(self._workers)
            for worker in workers:
                worker.terminate()
            self._stop_wait(3)

    def _stop_wait(self, timeout):
        start = time()
        while True:
            with self._task_lock:
                workers = list(self._workers)
            for worker in workers:
                if isinstance(worker, AsyncResult):
                    done = worker.ready()
                elif isinstance(worker, mp.Process):
                    worker.join(timeout=0)
                    done = not worker.is_alive()
                else:
                    raise RuntimeError(f"Unknown worker type: {type(worker).__name__}")

                if done:
                    with self._task_lock:
                        self._workers.remove(worker)
            with self._task_lock:
                if not self._workers:
                    return True
            if time() - start > timeout:
                return False
            sleep(0.01)


def test(filenames: list, ofile: str, verbose=False):
    predictor = get_test_predictor(verbose=verbose, workers=10)
    predictor.start()

    start = time()
    ids = []
    for i in range(100):
        ids.append(predictor.predict(filenames, ofile, wait=False))

    print("Waiting for results...")
    results = predictor.wait(ids)
    print(f"Finished predicting in {time()-start}")

    predictor.stop()
    pass

def get_test_predictor(verbose=False, workers=4, daemon=True):
    model = 'E:/output/results/tots/nnu_multi_v1-vertebrae-mean-max/_results/Dataset602_Bores/nnUNetTrainer_4000epochs_Batch2_NoMirroring__nnUNetPlans__2d'
    folds = (0,)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(tile_step_size=0.5,
                                use_gaussian=True,
                                use_mirroring=True,
                                perform_everything_on_device=True,
                                verbose=verbose)
    predictor.initialize_from_trained_model_folder(model, folds, 'checkpoint_best.pth')
    return ParallelPredictor(predictor, num_workers=workers, daemon=daemon)


if __name__ == '__main__':
    filenames = [f"E:/output/results/tots/nnu_multi_v1-all-mean-max/Dataset602_Bores/imagesTr/ct_s0000-fullorig_{i:04}.nrrd" for i in range(2)]
    ofile = r"E:/test/tots_profiling/predict_direct"
    test(filenames, ofile, verbose=False)
