import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal
import pandas as pd

from src.utils.dynamo_functions import delete_dynamo_enteries, convert_floats_to_decimal

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')  


def new_picks(table, keys, data):

    delete_dynamo_enteries(table, keys)

    for row in data:
        item = row.dict()
        item = convert_floats_to_decimal(item)
        table.put_item(Item=item)

