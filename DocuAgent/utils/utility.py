import requests



# Utility function to upload content to Vercel Blob Storage
def upload_to_vercel_blob(self, blob_path: str, content: str, content_type: str = "text/markdown") -> str:
        """
        Securely uploads data to Vercel Blob storage.
        """
        if not self.blob_token:
            raise ValueError("Cannot upload: Vercel Blob token is missing.")

        url = f"https://blob.vercel-storage.com/{blob_path}"
        headers = {
            "Authorization": f"Bearer {self.blob_token}",
            "x-api-version": "7",         
            "x-content-type": content_type, 
        }
        
        data = content.encode('utf-8')
        
        try:
            response = self.session.put(url, headers=headers, data=data, timeout=30)
            response.raise_for_status()
            
            print(f"Successfully uploaded to Vercel Blob: {blob_path}")
            return response.json().get("url")
            
        except requests.exceptions.RequestException as e:
            # Check if Vercel sent a specific error message back in the body
            error_details = response.text if response else "No response body"
            print(f"Vercel Blob upload failed for {blob_path}: {str(e)} | Details: {error_details}")
            raise RuntimeError(f"Failed to upload to Vercel Blob: {error_details}") from e