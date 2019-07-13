import boto3

ACCESS_KEY=' ... '
SECRET_ACCESS_KEY=' ... '

client = boto3.client(
	'sns',
	aws_access_key_id=ACCESS_KEY,
	aws_secret_access_key=SECRET_ACCESS_KEY,
	)

def publish(text):
	client.publish(TopicArn=' ... ',Message=text)
