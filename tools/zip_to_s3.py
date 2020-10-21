import boto3
import os

###
# To Use:
# 1. Download boto3
# 2. Fill in the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY fields
# 3. Create a folder named Data in the same directory as this file
# 4. Move the zip folders into Data
# 5. Run script
###

AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
BUCKET = "signsense-dev"
PATH = "./Data"

s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-2',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

if __name__ == "__main__":
    zips = os.listdir(PATH)
    for item in zips:
        s3.Bucket(BUCKET).upload_file(Filename=PATH+'/'+item, Key=item)
