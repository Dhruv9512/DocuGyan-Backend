import requests
from django.conf import settings
import re


from vercel.blob import BlobClient
from DocuAgent.models import DocuProcess, CustomUser
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection, MilvusClient



def sanitize_blob_filename(filename: str) -> str:
    name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
    name = name.strip()                            
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)   
    name = re.sub(r'_+', '_', name)                
    name = name.strip('_')                          
    return f"{name}.{ext}" if ext else name

def get_request_session_with_blob_auth():
    """
    Creates a requests.Session with Vercel Blob authentication if the token is available.
    This centralizes the logic for secure API calls to Vercel Blob storage.
    """
    session = requests.Session()
    blob_token = getattr(settings, 'VERCEL_BLOB_TOKEN', None)
    
    if blob_token:
        session.headers.update({"Authorization": f"Bearer {blob_token}"})
        print("Vercel Blob token found and added to session headers.")
    else:
        print("Warning: Vercel Blob token is missing. API calls to Blob storage will fail.")
    
    return session

def upload_to_vercel_blob(blob_path: str, content, content_type: str = "text/markdown") -> str:
    """
    Securely uploads data to Vercel Blob storage using the OFFICIAL Vercel Python SDK.
    """
    blob_token = getattr(settings, 'VERCEL_BLOB_TOKEN', None)
    if not blob_token:
        raise ValueError("Cannot upload: Vercel Blob token is missing.")

    # Safely convert strings to bytes for the HTTP payload
    data = content.encode('utf-8') if isinstance(content, str) else content
    
    # 1. Initialize the official client explicitly with your token
    client = BlobClient(token=blob_token)
    
    try:
        # 2. Upload using POSITIONAL arguments for the path and body
        blob = client.put(
            blob_path,                  # Positional Arg 1: The path
            data,                       # Positional Arg 2: The file content
            access="private",           # Keyword args for options
            content_type=content_type,
        )
        
        # 3. The SDK returns an object with a .url property
        print(f"Successfully uploaded to Vercel Blob: {blob.url}")
        return blob.url
        
    except Exception as e:
        print(f"Vercel Blob upload failed for {blob_path}: {str(e)}")
        raise RuntimeError(f"Failed to upload to Vercel Blob: {str(e)}") from e
         

def delete_collection_related_data(folder_name: str) -> None:
    """
    Securely deletes an entire folder, including ALL sub-folders and files, 
    from Vercel Blob storage using the OFFICIAL Vercel Python SDK.
    """
    blob_token = getattr(settings, 'VERCEL_BLOB_TOKEN', None)
    if not blob_token:
        raise ValueError("Cannot delete: Vercel Blob token is missing.")

    # Ensure trailing slash so the prefix matches the folder and everything inside it
    prefix = folder_name if folder_name.endswith('/') else f"{folder_name}/"
    client = BlobClient(token=blob_token)
    
    try:
        cursor = None
        has_more = True
        total_deleted = 0
        
        # 1. Loop to handle pagination in case the folder has hundreds/thousands of items
        while has_more:
            # Pass the cursor if one exists to get the next batch of files
            listing = client.list_objects(prefix=prefix, cursor=cursor)
            
            # Extract URLs
            urls_to_delete = [blob.url for blob in listing.blobs]
            
            # 2. Batch delete this chunk
            if urls_to_delete:
                client.delete(urls_to_delete)
                total_deleted += len(urls_to_delete)
                
            # 3. Check if there are more sub-folders/files left to fetch
            cursor = getattr(listing, 'cursor', None)
            has_more = bool(cursor)
            
        if total_deleted == 0:
            print(f"No blobs found in collection folder: {prefix}")
        else:
            print(f"Successfully deleted {total_deleted} items (including sub-folders) from {prefix}")
        
    except Exception as e:
        print(f"Vercel Blob deletion failed for folder {prefix}: {str(e)}")
        raise RuntimeError(f"Failed to delete folder from Vercel Blob: {str(e)}") from e
    
    try:
         # 3. Delete it from Milvus
        connections.connect(
            alias="default",
            uri=settings.ZILLIZ_URI,
            token=settings.ZILLIZ_TOKEN,
            secure=True,
            timeout=60
        )
        if utility.has_collection(folder_name):
            utility.drop_collection(folder_name, using="default")
            print(f"Successfully deleted Milvus collection: {folder_name}")
 
    except Exception as e:
        print(f"Milvus cleanup failed for collection {folder_name}: {str(e)}")
        raise RuntimeError(f"Failed to clean up Milvus collection: {str(e)}") from e

def get_collection_name(project_id) -> str:
    """
    Generates a safe collection name for Vector DBs by querying the database 
    for the user associated with the project_id.
    
    Format: collection_(email_prefix)_(project_id_last_4)
    """
    # Fetch the DocuProcess to get the user_uuid
    try:
        process = DocuProcess.objects.get(project_id=project_id)
    except DocuProcess.DoesNotExist:
        raise ValueError(f"Cannot generate collection name: No DocuProcess found for project_id {project_id}")

    # Fetch the User and derive a "username"
    try:
        if not process.user_uuid:
            raise ValueError("Process has no associated user_uuid")
            
        user = CustomUser.objects.get(user_uuid=process.user_uuid)
        
        raw_username = user.email.split('@')[0]
        
    except (CustomUser.DoesNotExist, ValueError):
        raw_username = "anonymous_user"

    # Get the last 4 characters of the project_id
    project_id_str = str(project_id).replace('-', '') 
    last_4_chars = project_id_str[-4:]
    
    # Sanitize the derived username (replace special chars with underscores, lowercase)
    safe_username = re.sub(r'[^a-zA-Z0-9]', '_', raw_username).lower()
    
    # Construct the final safe string
    collection_name = f"collection_{safe_username}_{last_4_chars}"
    
    return collection_name



def create_zilliz_collection(collection_name: str, dim: int, project_id: str) -> bool:
    """
    Creates a collection in Zilliz matching the specified Document schema.
    Uses ONLY the traditional PyMilvus ORM API to prevent mixing connection types.
    """
    try:
        print("Connecting to Zilliz via ORM...")
        
        # 1. Open ORM Connection securely
        connections.connect(
            alias="default",
            uri=settings.ZILLIZ_URI,
            token=settings.ZILLIZ_TOKEN,
            secure=True,
            timeout=60
        )
        
        # 2. Check existence
        if utility.has_collection(collection_name, using="default"):
            print(f"Collection '{collection_name}' already exists.")
            return False
            
        # 3. Enforce limit & cleanup
        existing_collections = utility.list_collections(using="default")
        
        if len(existing_collections) >= 5:
            print(f"Limit reached ({len(existing_collections)} collections). Cleaning up...")
            
            # Since ORM lacks a native timestamp fetcher, we drop the first available one 
            collection_to_drop = existing_collections[0]        
            # Clean up associated blobs and DB entries
            delete_collection_related_data(folder_name=collection_to_drop)
            print(f"Both Blob and milvus data cleanup complete for collection: {collection_to_drop}")
            DocuProcess.objects.filter(project_id=project_id, collection_name=collection_to_drop).delete()
        
        # 4. Create the Schema using ORM
        print(f"Creating custom schema for '{collection_name}'...")
        
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="page_content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="project_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="source_title", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="is_vision_extracted", dtype=DataType.BOOL),
            FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="question_text", dtype=DataType.VARCHAR, max_length=1000, nullable=True),
        ]
        
        schema = CollectionSchema(
            fields=fields, 
            description="Schema for LangChain Document ingestion"
        )

        # 5. Create the Collection
        collection = Collection(
            name=collection_name, 
            schema=schema, 
            using="default", 
            consistency_level="Session"
        )
        
        # 6. Define & Create all Indexes
        vector_index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200}
        }
        
        # Build Vector Index
        collection.create_index(field_name="vector", index_params=vector_index_params, using="default")
        
        # Build Scalar Indexes for fast metadata filtering
        collection.create_index(field_name="project_id", index_params={"index_type": "STL_SORT"}, using="default")
        collection.create_index(field_name="source_url", index_params={"index_type": "STL_SORT"}, using="default")
        collection.create_index(field_name="chunk_type", index_params={"index_type": "STL_SORT"}, using="default")

        print(f"Collection '{collection_name}' successfully created with all indexes via ORM.")
        return True

    except Exception as e:
        print(f"Critical error during collection setup: {e}")
        return False

