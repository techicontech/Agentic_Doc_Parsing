"""S3-compatible MinIO client for figure/page images."""

from __future__ import annotations

import io
from typing import BinaryIO

import boto3
from botocore.client import Config

from marine_docs.config import get_settings


def get_s3_client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=("https://" if s.minio_secure else "http://") + s.minio_endpoint,
        aws_access_key_id=s.minio_access_key,
        aws_secret_access_key=s.minio_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket() -> str:
    s = get_settings()
    client = get_s3_client()
    buckets = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
    if s.minio_bucket not in buckets:
        client.create_bucket(Bucket=s.minio_bucket)
    return s.minio_bucket


def upload_bytes(key: str, data: bytes, content_type: str = "image/png") -> str:
    bucket = ensure_bucket()
    client = get_s3_client()
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    return key


def upload_fileobj(key: str, fileobj: BinaryIO, content_type: str = "image/png") -> str:
    bucket = ensure_bucket()
    client = get_s3_client()
    client.upload_fileobj(
        fileobj,
        bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return key


def upload_png(key: str, png_bytes: bytes) -> str:
    return upload_bytes(key, png_bytes, content_type="image/png")


def download_bytes(key: str) -> bytes:
    bucket = ensure_bucket()
    client = get_s3_client()
    obj = client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def bytes_buffer(data: bytes) -> io.BytesIO:
    return io.BytesIO(data)
