from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = FastAPI()

def read_csv_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    print(f"Loaded {len(df)} rows of data. Date range: {df.index.min()} to {df.index.max()}")
    return df

data = read_csv_data('sample_energy_data.csv')

def predict_next_hour(series, timestamp):
    two_day_data = series.loc[timestamp - timedelta(days=2):timestamp]
    return two_day_data.groupby(two_day_data.index.hour).mean()[timestamp.hour]

@app.get("/api/predict/consumption")
async def predict_consumption(timestamp: str = Query(None, description="Timestamp in format YYYY-MM-DD HH:MM:SS")):
    global data
    
    if timestamp:
        input_time = pd.to_datetime(timestamp)
        if input_time not in data.index:
            raise HTTPException(status_code=400, detail="Timestamp not found in data")
        sample = data.loc[input_time]
    else:
        sample = data.iloc[-1]
    
    predicted_consumption = predict_next_hour(data['energy_consumption'], sample.name)
    
    return {
        'timestamp': sample.name.strftime('%Y-%m-%d %H:%M:%S'),
        'actual_consumption': float(sample['energy_consumption']),
        'predicted_consumption': float(predicted_consumption),
        'prediction_for': (sample.name + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)