import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal
import pandas as pd

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')  

def delete_dynamo_enteries(table, keys):

    partition_key=keys['PartitionKey']
    sort_key=keys['SortKey']

    response = table.scan()
    items = response['Items']
    with table.batch_writer() as batch:
        for item in items:
            batch.delete_item(Key={
                partition_key: item[partition_key],
                sort_key: item[sort_key]
            })

def convert_floats_to_decimal(item: dict) -> dict:
    for key, value in item.items():
        if isinstance(value, float):
            item[key] = Decimal(str(value))  # Convert float to Decimal
    return item

def new_picks(table, keys, data):

    delete_dynamo_enteries(table, keys)

    for row in data:
        item = row.dict()
        item = convert_floats_to_decimal(item)
        table.put_item(Item=item)

