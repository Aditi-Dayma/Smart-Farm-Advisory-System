"""
AWS Lambda Function: Pesticide Advisory Rule Engine & Dispatcher.

Receives event payload with disease detection findings (Module 2 output),
evaluates weather constraints (wind, rain, temp), performs official lookup,
stores recommendation in DynamoDB, and alerts the farmer via SNS.
"""

import json
import os
from datetime import datetime
import boto3

DYNAMODB_TABLE_PESTICIDES = os.environ.get("TABLE_PESTICIDE_ADVISORIES", "PesticideAdvisories")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")

dynamodb = boto3.resource("dynamodb")
sns_client = boto3.client("sns")

DISCLAIMER_TEXT = (
    "⚠️ DISCLAIMER: This is decision support only and not a replacement for a certified agronomist. "
    "Verify product labels, dilution rates, and safety intervals before application."
)


def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
    except Exception:
        body = event

    farm_id = body.get("farm_id", "default_farm")
    crop = body.get("crop", "Tomato")
    disease = body.get("disease", "Early Blight")
    confidence = float(body.get("confidence", 0.90))
    wind_speed = float(body.get("wind_speed", 5.0))
    rain_forecast = float(body.get("rain_forecast", 0.0))
    temperature = float(body.get("temperature", 28.0))

    # Governance rule checks
    safe_to_spray = True
    timing_reasons = []

    if wind_speed > 15.0:
        safe_to_spray = False
        timing_reasons.append(f"High wind ({wind_speed} km/h > 15 km/h) risk of spray drift")
    if rain_forecast > 1.0:
        safe_to_spray = False
        timing_reasons.append(f"Rain forecast ({rain_forecast} mm) will wash off chemical")
    if temperature > 35.0:
        safe_to_spray = False
        timing_reasons.append(f"Temperature high ({temperature}°C > 35°C) phytotoxicity risk")

    timing_summary = "; ".join(timing_reasons) if timing_reasons else "Conditions safe for spray (Morning 6-8 AM / Evening 4-6 PM)"

    # Look up treatment (Sample representation from Module 4 database)
    treatment_recommendation = {
        "chemical_option": "Mancozeb 75% WP @ 2.5 g/L water (625 g/ha)",
        "biocontrol_option": "Trichoderma viride 1% WP @ 5 g/L water",
        "spray_interval_days": 7,
        "pre_harvest_interval_days": 5,
        "safety_ppe": ["Gloves", "N95 Mask", "Protective Eyewear"],
    }

    timestamp_str = datetime.utcnow().isoformat()
    advisory_record = {
        "farm_id": farm_id,
        "timestamp": timestamp_str,
        "crop": crop,
        "disease": disease,
        "confidence": str(round(confidence, 3)),
        "safe_to_spray": safe_to_spray,
        "timing_reason": timing_summary,
        "treatments": treatment_recommendation,
        "disclaimer": DISCLAIMER_TEXT,
    }

    # Store in DynamoDB
    table = dynamodb.Table(DYNAMODB_TABLE_PESTICIDES)
    table.put_item(Item=advisory_record)

    # Dispatch SNS alert
    if SNS_TOPIC_ARN:
        msg = (
            f"🔬 Smart Farm Pest Alert for Farm {farm_id}:\n"
            f"Detected: {disease} in {crop} (Confidence: {int(confidence*100)}%).\n"
            f"Status: {'SAFE TO SPRAY' if safe_to_spray else 'DO NOT SPRAY NOW'}.\n"
            f"Reason: {timing_summary}\n"
            f"Treatment: {treatment_recommendation['chemical_option']}\n\n"
            f"{DISCLAIMER_TEXT}"
        )
        try:
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=msg,
                Subject=f"Pest Advisory: {crop} - {disease}",
            )
        except Exception as e:
            print(f"Failed to publish SNS: {e}")

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps(advisory_record),
    }
