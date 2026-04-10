import requests
from django.conf import settings
import re


from vercel.blob import BlobClient
from DocuAgent.models import DocuProcess, CustomUser
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection


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


def create_zilliz_collection(collection_name: str, dim: int ) -> bool:
    """
    Creates a collection in Zilliz matching the specified Document schema.
    """
    try:
        # 1. Connect to Zilliz Cloud
        connections.connect(
            alias=settings.ZILLIZ_ALIAS,
            uri=settings.ZILLIZ_URI,
            token=settings.ZILLIZ_TOKEN
        )
        print(f"Connected to Zilliz at URI")

        # 2. Check if collection already exists
        try:
            if utility.has_collection(collection_name, using="default"):
                print(f"Collection '{collection_name}' already exists.")
                return False
        except Exception as e:
            print(f"Error checking collection existence: {e}")
            return False
            
        print(f"Creating custom schema for '{collection_name}'...")
        
        # 3. Define fields based on your Document payload
        fields = [
            # Primary Key & Vector
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            
            # Document Content
            FieldSchema(name="page_content", dtype=DataType.VARCHAR, max_length=65535),
            
            # Metadata Mapping
            FieldSchema(name="project_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="source_title", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="is_vision_extracted", dtype=DataType.BOOL),
        ]
        
        schema = CollectionSchema(
            fields=fields, 
            description="Schema for LangChain Document ingestion"
        )

        # 4. Create collection
        try:
            collection = Collection(
                name=collection_name, 
                schema=schema, 
                consistency_level='Session', 
                using="default"
            )
            print(f"Collection '{collection_name}' successfully created.")
        except Exception as e:
            print(f"Failed to create collection '{collection_name}': {str(e)}")
            return False
        
        # 5. Create Vector Index
        vector_index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index(field_name="vector", index_params=vector_index_params)
        print("Vector index (HNSW/COSINE) created.")

        # 6. Create Scalar Indexes for faster metadata filtering
        collection.create_index(field_name="project_id", index_params={"index_type": "STL_SORT"})
        collection.create_index(field_name="source_url", index_params={"index_type": "STL_SORT"})
        print("Scalar indexes on 'project_id' and 'source_url' created.")
        
        return True

    except Exception as e:
        print(f"Critical error during collection setup: {e}")
        return False