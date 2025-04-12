from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import os

# Initialize app
app = FastAPI()

# Load model
model = joblib.load(r"C:\Users\USER\UM\xgboost_boost_model_flexible.pkl")

# Load dataset
df = pd.read_excel(r"C:\Users\USER\UM\merchant_sales_allhours.csv")

# Mount static (optional)
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")


# Route for the HTML page
@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


# Endpoint to provide item dropdown list
@app.get("/items")
def get_item_list():
    grouped = df.groupby("item_id").first().reset_index()
    results = []
    for _, row in grouped.iterrows():
        results.append({
            "item_id": int(row["item_id"]),
            "item_name": row["item_name"],
            "sales": float(row["sales"]),
            "average": float(row["average"]),
            "cuisine_id": int(row["cuisine_id"])
        })
    return results


# Input model for simplified prediction
class LiteInput(BaseModel):
    item_id: int
    sales: float
    average: float
    cuisine_id: int


# Main prediction logic
@app.post("/predict_all/")
def predict_across_time_blocks(data: LiteInput):
    try:
        # Main prediction
        sales_gt_avg = 1 if data.sales > data.average else 0
        main_features = pd.DataFrame([{
            "item_id": data.item_id,
            "sales": data.sales,
            "average": data.average,
            "sales_gt_avg": sales_gt_avg
        }])
        main_prediction = int(model.predict(main_features)[0])

        # All time blocks for this item_id
        matching_rows = df[df['item_id'] == data.item_id]
        results: List[dict] = []

        for _, row in matching_rows.iterrows():
            row_sales_gt_avg = 1 if row["sales"] > row["average"] else 0
            input_df = pd.DataFrame([{
                "item_id": row["item_id"],
                "sales": row["sales"],
                "average": row["average"],
                "sales_gt_avg": row_sales_gt_avg
            }])
            prediction = int(model.predict(input_df)[0])
            results.append({
                "time_block": int(row["time_block"]),
                "prediction": prediction

            })

        return {
            "main_prediction": main_prediction,
            "time_block_predictions": results
        }

    except Exception as e:
        return {"error": str(e)}
