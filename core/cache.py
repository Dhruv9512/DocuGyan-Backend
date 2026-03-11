from django.core.cache import cache

def delete_all_user_cache(user):
    """Utility to clear cached data upon logout."""
    if not user or user.is_anonymous:
        return
    
    # Clears the entire cache for the whole application
    cache.clear()