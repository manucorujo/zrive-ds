import boto3
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)


def load_variables():
    load_dotenv()

    access_key = os.getenv("ACCESS_KEY_ID")
    secret_key = os.getenv("SECRET_ACCESS_KEY")
    bucket_name = os.getenv("BUCKET_NAME")
    s3_directory = "groceries/sampled-datasets/"
    local_directory = "/home/manucorujo/zrive-ds-data/"

    return {
        "access_key": access_key,
        "secret_key": secret_key,
        "bucket_name": bucket_name,
        "s3_directory": s3_directory,
        "local_directory": local_directory,
    }


def download_s3_data(
    access_key: str,
    secret_key: str,
    bucket_name: str,
    s3_directory: str,
    local_directory: str,
):
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

    s3 = boto3.client(
        "s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )

    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_directory)

        if "Contents" in response:
            for obj in response["Contents"]:
                file_name = obj["Key"]
                if file_name.endswith("/"):
                    continue

                local_file_name = os.path.join(
                    local_directory, os.path.basename(file_name)
                )

                s3.download_file(bucket_name, file_name, local_file_name)
                logging.info(f"Downloaded: {file_name} -> {local_file_name}")
        else:
            logging.info(f"No files found in the directory: {s3_directory}")

    except Exception as e:
        logging.error(f"Error downloading files: {e}")


def main():
    variables = load_variables()
    download_s3_data(
        variables["access_key"],
        variables["secret_key"],
        variables["bucket_name"],
        variables["s3_directory"],
        variables["local_directory"],
    )


if __name__ == "__main__":
    main()
