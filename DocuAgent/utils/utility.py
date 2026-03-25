import requests
from django.conf import settings

def upload_to_vercel_blob(blob_path: str, content: str, content_type: str = "text/markdown") -> str:
    """
    Securely uploads data to Vercel Blob storage as a standalone utility.
    """
    # 1. Get the token directly from settings
    blob_token = getattr(settings, 'VERCEL_BLOB_TOKEN', None)
    
    if not blob_token:
        raise ValueError("Cannot upload: Vercel Blob token is missing.")

    url = f"https://blob.vercel-storage.com/{blob_path}"
    headers = {
        "Authorization": f"Bearer {blob_token}",
        "x-api-version": "7",         
        "x-content-type": content_type, 
    }
    
    data = content.encode('utf-8')
    response = None # Prevent UnboundLocalError if the request times out
    
    try:
        response = requests.put(url, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        
        print(f"Successfully uploaded to Vercel Blob: {blob_path}")
        return response.json().get("url")
        
    except requests.exceptions.RequestException as e:
        error_details = response.text if response else "No response body"
        print(f"Vercel Blob upload failed for {blob_path}: {str(e)} | Details: {error_details}")
        raise RuntimeError(f"Failed to upload to Vercel Blob: {error_details}") from e