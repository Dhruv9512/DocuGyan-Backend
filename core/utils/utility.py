from DocuAgent.models import DocuProcess, CustomUser
import re


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
