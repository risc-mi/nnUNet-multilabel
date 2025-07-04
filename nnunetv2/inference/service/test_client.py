import multiprocessing
import os
from tempfile import TemporaryFile, TemporaryDirectory
from time import time

import requests
import SimpleITK as sitk

server_url = "http://localhost:8000"

def predict_server(image_path, ofile):
    response = requests.post(
        url=f"{server_url}/predict",
        json={
            "filenames": image_path,
            "ofile": ofile
        }
    )
    if response.status_code == 200:
        if 'json' in response.headers['Content-Type']:
            data = response.json()
            if data["success"]:
                result = f"Image path: {data["result"]}"
            else:
                result = f"Prediction failed: {data}"
        elif 'octet-stream' in response.headers['Content-Type']:
            with TemporaryDirectory() as tmp:
                fn = os.path.join(tmp, "result.nrrd")
                with open(fn, 'wb') as f:
                    f.write(response.content)
                img = sitk.ReadImage(fn)
                result = f"Image of size {img.GetSize()}"
        else:
            raise RuntimeError("Unknown response type:", response.headers['Content-Type'])
    else:
        result = f"Request failed: {response.status_code}"
    return result

def run_prediction(id):
    filenames = [f"E:/output/results/tots/nnu_multi_v1-all-mean-max/Dataset602_Bores/imagesTr/ct_s0000-fullorig_{idx:04}.nrrd" for idx in range(2)]
    ofile = f"E:/test/tots_profiling/predict_direct_{id:03}.nrrd"


    start = time()
    result = predict_server(filenames, ofile)
    print("Elapsed:", time() - start)

    print(f"Result {id}: {result}")

if __name__ == '__main__':
    #for i in range(100):
    #    run_prediction(0)
    start = time()
    with multiprocessing.Pool(processes=10) as pool:
        pool.map(run_prediction, range(100))
    print("Total time:", time() - start)