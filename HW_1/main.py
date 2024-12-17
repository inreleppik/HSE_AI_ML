
import asyncio
import uvicorn
import pandas as pd
import numpy as np
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from c_transformer import CustomCarDataTransformer 
import joblib

app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)

model = joblib.load('models/ln_lr_cat_pipe.pkl')



class Item(BaseModel):
    name: str = Field(..., example= "Mahindra Xylo E4 BS IV")
    year: int = Field(..., example = 2010)
    selling_price: int = Field(..., example = 229999)
    km_driven: int = Field(..., example = 168000)
    fuel: str = Field(..., example = "Diesel")
    seller_type: str = Field(..., example = "Individual")
    transmission: str = Field(..., example = "Manual")
    owner: str = Field(..., example = "First Owner")
    mileage: str = Field(..., example = "14.0 kmpl")
    engine: str = Field(..., example = "2498 CC")
    max_power: str = Field(..., example = "112 bhp")
    torque: str = Field(..., example = "260 Nm at 1800-2200 rpm")
    seats: float = Field(..., example = 7.0)


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame([item.model_dump()]).drop(columns=['selling_price'])
    return np.exp(model.predict(data))


@app.post("/predict_items")
def predict_items(file: UploadFile = File()) -> StreamingResponse:
    data = pd.read_csv(file.file)
    
    data_for_predict = data.drop(columns=['selling_price']).copy()
    preds = np.exp(model.predict(data_for_predict))

    data['price_preds'] = preds

    output = io.StringIO()
    data.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        output, media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=df_with_preds_{file.filename}"}
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
