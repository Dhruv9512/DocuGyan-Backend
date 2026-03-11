from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

@shared_task(queue='DocuMail_tasks')
def Otp_Verification(data):
    """Sends OTP for login/signup."""
    try:
        email = data.get("email")
        otp = data.get("otp")
        subject = "DocuGyan - Your Login OTP"
        message = f"Your One Time Password (OTP) for DocuGyan is: {otp}\nThis OTP is valid for a short time. Please do not share it with anyone."
        
        # Anymail automatically intercepts this and sends it via the Brevo API!
        send_mail(
            subject=subject,
            message=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email],
            fail_silently=False,
        )
        return f"OTP sent to {email}"
    except Exception as e:
        logger.error(f"Failed to send OTP to {email}: {str(e)}")
        raise e

@shared_task(queue='DocuMail_tasks')
def send_login_success_email(data):
    """Sends a welcome/login success email."""
    try:
        email = data.get("email")
        first_name = data.get("first_name", "User")
        subject = "Welcome to DocuGyan"
        message = f"Hi {first_name},\n\nYou have successfully logged into DocuGyan. Welcome aboard!"
        
        # Anymail automatically intercepts this too
        send_mail(
            subject=subject,
            message=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email],
            fail_silently=False,
        )
        return f"Success email sent to {email}"
    except Exception as e:
        logger.error(f"Failed to send success email to {email}: {str(e)}")
        raise e