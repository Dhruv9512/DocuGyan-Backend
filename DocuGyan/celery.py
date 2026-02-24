import os
from celery import Celery
from django.conf import settings

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DocuGyan.settings')

# Initialize Celery app for DocuGyan
app = Celery('DocuGyan', broker=settings.CELERY_BROKER_URL)

app.config_from_object('django.conf:settings', namespace='CELERY')

# Update configuration to retry connecting to the broker on startup
app.conf.update(
    broker_connection_retry_on_startup=True,
)

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

def stop_task(task_id, is_forced=False):
    """
    Utility function to revoke a running Celery task.
    Highly useful for cancelling document processing workflows.
    """
    try:
        signal = 'SIGKILL' if is_forced else 'SIGTERM'
        
        if isinstance(task_id, str):
            app.control.revoke(task_id, terminate=True, signal=signal)
        else:
            task_id_ = task_id.decode('utf-8')
            app.control.revoke(task_id_, terminate=True, signal=signal)
        return True
    except Exception as err:
        print(f'Error occurred while killing the taskID: {task_id} due to {err}')
        return False