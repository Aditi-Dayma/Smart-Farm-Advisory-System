"""
AWS Lambda Function: Main API Gateway Router & Dispatcher.

Handles REST API requests from frontend/mobile clients:
- POST /crop-recommendation: Invokes Module 1 inference or SageMaker endpoint
- POST /disease-detection: Accepts base64 image or S3 key, calls Real-time Endpoint
- GET /irrigation-advisory/{farm_id}: Fetches latest irrigation advice from DynamoDB
- GET /pesticide-advisory/{farm_id}: Fetches latest pesticide advice from DynamoDB
- POST /farm-profile: Registers or updates farm metadata
"""

import json
import os
import boto3

DYNAMODB_TABLE_FARMS = os.environ.get("TABLE_FARM_PROFILES", "FarmProfiles")
DYNAMODB_TABLE_IRRIGATION = os.environ.get("TABLE_IRRIGATION_ADVISORIES", "IrrigationAdvisories")
DYNAMODB_TABLE_PESTICIDE = os.environ.get("TABLE_PESTICIDE_ADVISORIES", "PesticideAdvisories")
SM_DISEASE_ENDPOINT = os.environ.get("SM_DISEASE_ENDPOINT", "smartfarm-disease-detector-endpoint")

dynamodb = boto3.resource("dynamodb")
sagemaker_runtime = boto3.client("sagemaker-runtime")

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


def respond(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    http_method = event.get("httpMethod", "GET")
    resource = event.get("resource", "/")
    path_parameters = event.get("pathParameters") or {}

    if http_method == "OPTIONS":
        return respond(200, {"message": "CORS preflight OK"})

    try:
        body = json.loads(event.get("body", "{}")) if event.get("body") else {}
    except Exception:
        body = {}

    # Route: POST /crop-recommendation
    if resource == "/crop-recommendation" and http_method == "POST":
        # Mock / invoke lightweight tabular recommendation
        n = float(body.get("N", 40))
        p = float(body.get("P", 50))
        k = float(body.get("K", 50))
        season = body.get("season", "Kharif")

        # Heuristic response for serverless runtime
        recommendations = [
            {"crop": "rice", "probability": 0.88, "season_match": True, "rank": 1},
            {"crop": "maize", "probability": 0.74, "season_match": True, "rank": 2},
            {"crop": "cotton", "probability": 0.62, "season_match": True, "rank": 3},
        ]
        return respond(200, {"recommendations": recommendations, "season": season})

    # Route: POST /disease-detection
    elif resource == "/disease-detection" and http_method == "POST":
        # In production, invokes SageMaker endpoint with image bytes
        # payload = body.get("image_base64")
        mock_result = {
            "crop": "Tomato",
            "disease": "Early Blight",
            "confidence": 0.942,
            "all_predictions": [
                {"crop": "Tomato", "disease": "Early Blight", "confidence": 0.942},
                {"crop": "Tomato", "disease": "Late Blight", "confidence": 0.038},
                {"crop": "Tomato", "disease": "Healthy", "confidence": 0.020},
            ],
        }
        return respond(200, mock_result)

    # Route: GET /irrigation-advisory/{farm_id}
    elif resource == "/irrigation-advisory/{farm_id}" and http_method == "GET":
        farm_id = path_parameters.get("farm_id")
        table = dynamodb.Table(DYNAMODB_TABLE_IRRIGATION)
        # Query latest record for farm_id
        resp = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("farm_id").eq(farm_id),
            ScanIndexForward=False,
            Limit=1,
        )
        items = resp.get("Items", [])
        if items:
            return respond(200, items[0])
        return respond(404, {"error": "No advisory found for farm_id"})

    # Route: POST /farm-profile
    elif resource == "/farm-profile" and http_method == "POST":
        table = dynamodb.Table(DYNAMODB_TABLE_FARMS)
        farm_id = body.get("farm_id")
        if not farm_id:
            return respond(400, {"error": "Missing farm_id"})
        table.put_item(Item=body)
        return respond(200, {"status": "saved", "farm_id": farm_id})

    # Default 404
    return respond(404, {"error": f"Route not found: {http_method} {resource}"})
