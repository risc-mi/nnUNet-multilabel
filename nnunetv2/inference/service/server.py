import asyncio
import os
import sys
import tempfile
import traceback
from io import BytesIO
from typing import List, Optional

from fastapi import FastAPI, UploadFile, HTTPException
import uuid
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from nnunetv2.inference.service.predictor import get_test_predictor

app = FastAPI(lifespan=lambda _: predictor_lifespan())
predictor = get_test_predictor(workers=10)
APP_DATA = None

@asynccontextmanager
async def predictor_lifespan():
    """
    Custom lifespan context manager to initialize the predictor.
    """
    global TEMP_DIR
    with tempfile.TemporaryDirectory() as temp:
        TEMP_DIR = temp

        # Initialize the predictor
        predictor.start()
        print("Prediction server started.")

        try:
            yield
        finally:
            print("Shutting down the prediction server...")
            try:
                predictor.shutdown()
            except Exception:
                print("Failed to shutdown the prediction server.", file=sys.stderr)
                traceback.print_exc()




class PredictRequest(BaseModel):
    filenames: List[str] = Field(..., description="List of file paths to predict")
    ofile: Optional[str] = Field(None, description="File or folder to write the output to")

@app.post("/predict")
async def predict(request: PredictRequest):
    filenames = request.filenames
    ofile = request.ofile
    if filenames:
        tmp_filenames = None
        tmp_ofile = None
        try:
            id = uuid.uuid4()
            print(f"Request {id} started")

            if isinstance(filenames, UploadFile):
                tmp_filenames = os.path.join(TEMP_DIR, 'inputs', f"{id.hex}.nrrd")
                os.makedirs(os.path.dirname(tmp_filenames), exist_ok=True)
                with open(tmp_filenames, "wb") as temp_file:
                    temp_file.write(await filenames.read())
                filenames = tmp_filenames

            if isinstance(filenames, str):
                filenames = [filenames]

            if not all(isinstance(file, str) and os.path.exists(file) for file in filenames):
                raise HTTPException(status_code=400, detail="Invalid file path provided")

            if ofile is None:
                tmp_ofile = ofile = os.path.join(TEMP_DIR, 'outputs', f"{id.hex}.nrrd")

            #async def _run_async(future):
            #    task_id = predictor.predict(id=id, filenames=filenames, ofile=ofile, wait=False)
            #    result = None
            #    while result is None:
            #        result = await predictor.wait(task_id, timeout=0)

            #    future.set_result(result)

            #future = asyncio.Future()
            #asyncio.create_task(_run_async(future))
            #result = await future
            result = predictor.predict(id=id, filenames=filenames, ofile=ofile, wait=True)

            if result is None:
                return {"success": False, "error": "Unknown error occurred, no result returned"}
            if result.success:
                if tmp_ofile:
                    with open(tmp_ofile, 'rb') as f:
                        ofile_bytes = BytesIO(f.read())
                    return StreamingResponse(ofile_bytes, media_type="application/octet-stream",
                                             headers={"Content-Disposition": f"attachment; filename={os.path.basename(result.ofile)}"})
                else:
                    return {"success": True, "result": result.ofile}
            else:
                return {"success": False, "error": str(result.error)}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error starting prediction: {str(e)}")
        finally:
            for tmp in [tmp_filenames, tmp_ofile]:
                if tmp is not None:
                    os.remove(tmp)
            print(f"Request {id} ended")

    else:
        raise HTTPException(status_code=400, detail="No file or path provided")


@app.get("/health")
async def health_check():
    """
    A health check endpoint to ensure the service is running.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)