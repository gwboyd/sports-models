import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal
import pandas as pd

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')  

def convert_to_decimal(value):
    if isinstance(value, float):
        return Decimal(str(value))
    return value

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

            
def dynamo_add_column(table, df, column_name, new_write_function, delete_function, keys):

    saved_table = pd.DataFrame(table.scan().get('Items'))
    
    if len(saved_table)>0:
        print(f"{len(saved_table)} rows saved. DF has {saved_table.shape[1]} columns")
    
    delete_function(table, keys)
    print("Table deleted")

    saved_table = saved_table.merge(df[['game_id', column_name]], on='game_id', how='left')
    print(f"Now {len(saved_table)} rows and {saved_table.shape[1]} columns")

    new_write_function(saved_table, table)
    updated_dynamo_table = pd.DataFrame(table.scan().get('Items'))
    print(f"Write done, updated dynamo table has {len(updated_dynamo_table)} rows and {updated_dynamo_table.shape[1]} columns")
    

def query_results(table, key_object):
    key_name = key_object[0]
    key_val = key_object[1]

    response = table.query(
        KeyConditionExpression=Key(key_name).eq(key_val)
    )
    return response.get('Items')

def scan_table(table):

    response = table.scan()

    return response.get('Items', [])


    