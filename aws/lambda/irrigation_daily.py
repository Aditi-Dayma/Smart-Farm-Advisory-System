"""
AWS Lambda Function: Daily Irrigation Advisory Scheduler.

Triggered daily at 06:00 AM IST (00:30 UTC) by Amazon EventBridge.
Workflow:
1. Scan DynamoDB FarmProfiles table for active farm subscriptions.
2. For each farm:
   a. Query Open-Meteo weather API for local 24h precipitation and ET0.
   b. Calculate water balance deficit (or invoke SageMaker Batch/Endpoint).
   c. Determine if irrigation is needed and recommended water volume (liters).
3. Persist generated advisory into DynamoDB IrrigationAdvisories table.
4. If irrigation is urgently required or severe weather is forecast, publish an alert via Amazon SNS.
"""

import json
import os
from datetime import datetime
import urllib.request
import boto3

DYNAMODB_TABLE_FARMS = os.environ.get("TABLE_FARM_PROFILES", "FarmProfiles")
DYNAMODB_TABLE_ADVISORIES = os.environ.get("TABLE_IRRIGATION_ADVISORIES", "IrrigationAdvisories")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")

dynamodb = boto3.resource("dynamodb")
sns_client = boto3.client("sns")


def fetch_weather(lat: float, lon: float) -> dict:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum,et0_fao_evapotranspiration,temperature_2m_max&timezone=auto"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SmartFarmAWS/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            daily = data.get("daily", {})
            return {
                "rain_today": daily.get("precipitation_sum", [0.0])[0],
                "et0": daily.get("et0_fao_evapotranspiration", [4.0])[0],
                "temp_max": daily.get("temperature_2m_max", [30.0])[0],
            }
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return {"rain_today": 0.0, "et0": 4.0, "temp_max": 30.0}


def lambda_handler(event, context):
    print("Initiating daily irrigation advisory run...")
    farms_table = dynamodb.Table(DYNAMODB_TABLE_FARMS)
    advisories_table = dynamodb.Table(DYNAMODB_TABLE_ADVISORIES)
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    # Scan registered farms
    resp = farms_table.scan()
    farms = resp.get("Items", [])
    processed_count = 0

    for farm in farms:
        farm_id = farm.get("farm_id")
        lat = float(farm.get("latitude", 19.07))
        lon = float(farm.get("longitude", 72.87))
        crop = farm.get("crop", "rice")
        area_ha = float(farm.get("area_hectares", 1.0))
        phone_number = farm.get("phone_number")

        weather = fetch_weather(lat, lon)
        et0 = weather["et0"]
        rain = weather["rain_today"]

        # Simple FAO-56 crop factor Kc heuristic
        kc = 1.1 if crop.lower() == "rice" else 0.8
        etc = et0 * kc
        net_deficit_mm = max(0.0, etc - rain)
        should_irrigate = net_deficit_mm > 5.0 and rain < 10.0

        amount_mm = net_deficit_mm if should_irrigate else 0.0
        volume_liters = amount_mm * area_ha * 10000

        advisory_item = {
            "farm_id": farm_id,
            "date": today_str,
            "should_irrigate": should_irrigate,
            "amount_mm": str(round(amount_mm, 2)),
            "volume_liters": str(round(volume_liters, 0)),
            "et0": str(round(et0, 2)),
            "etc": str(round(etc, 2)),
            "rain_forecast": str(round(rain, 2)),
            "created_at": datetime.utcnow().isoformat(),
        }

        # Store in DynamoDB
        advisories_table.put_item(Item=advisory_item)
        processed_count += 1

        # Send SNS alert if irrigation is recommended
        if should_irrigate and SNS_TOPIC_ARN:
            message = (
                f"🌾 Smart Farm Alert: Irrigation needed for {crop} today ({today_str}).\n"
                f"Recommended water: {round(amount_mm, 1)} mm ({int(volume_liters)} Liters for {area_ha} ha).\n"
                f"Rain forecast: {rain} mm. ETc: {round(etc, 1)} mm."
            )
            try:
                sns_client.publish(
                    TopicArn=SNS_TOPIC_ARN,
                    Message=message,
                    Subject=f"Irrigation Advisory - Farm {farm_id}",
                )
            except Exception as err:
                print(f"SNS publish error: {err}")

    return {
        "statusCode": 200,
        "body": json.dumps({"status": "success", "processed_farms": processed_count}),
    }
