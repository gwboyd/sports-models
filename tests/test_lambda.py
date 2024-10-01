from fastapi.testclient import TestClient
from mangum import Mangum
from main import app  # Import your FastAPI app
import json

handler = Mangum(app)

def test_lambda_handler():
    # event = {
    #     "httpMethod": "GET",
    #     "path": "/nfl-picks",
    #     "headers": {},
    #     "queryStringParameters": {},
    #     "requestContext": {
    #         "http": {
    #             "method": "GET",
    #             "path": "/nfl-picks"
    #         }
    #     }
    # }

    event = {
        "resource": "/nfl-picks",
        "path": "/nfl-picks",
        "httpMethod": "GET",
        "headers": {
            "Authorization": "jake"
        },
        "queryStringParameters": {},
        "multiValueQueryStringParameters": None,
        "pathParameters": None,
        "stageVariables": None,
        "requestContext": {
            "resourcePath": "/nfl-picks",
            "httpMethod": "GET",
            "path": "/nfl-picks",
            "identity": {
                "sourceIp": "127.0.0.1",
                "userAgent": "Mozilla/5.0"
            }
        },
        "body": None,
        "isBase64Encoded": False
    }


    

    context = {}  # Mock Lambda context if needed
    response = handler(event, context)

    # Parse the response body (it will be a string in Lambda)
    print("Response:", response)
    body = json.loads(response['body'])
    print("Body:", body)

    assert response['statusCode'] == 200


