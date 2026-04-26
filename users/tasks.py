from celery import shared_task
from django.core.mail import EmailMultiAlternatives
from django.conf import settings
from django.template.loader import render_to_string
from django.utils.html import strip_tags
import logging

logger = logging.getLogger(__name__)


def _send_html_email(subject, to_email, template_name, context):
    """Send multipart email with plain-text fallback and HTML template."""
    html_content = render_to_string(template_name, context)
    text_content = strip_tags(html_content)

    email_message = EmailMultiAlternatives(
        subject=subject,
        body=text_content,
        from_email=settings.DEFAULT_FROM_EMAIL,
        to=[to_email],
    )
    email_message.attach_alternative(html_content, "text/html")
    email_message.send(fail_silently=False)


@shared_task(queue='DocuMail_tasks')
def Otp_Verification(data):
    """Sends OTP for login/signup."""
    try:
        email = data.get("email")
        otp = data.get("otp")
        first_name = data.get("first_name", "User")
        subject = "DocuGyan - Your Login OTP"

        _send_html_email(
            subject=subject,
            to_email=email,
            template_name="users/emails/otp_verification.html",
            context={
                "first_name": first_name,
                "otp": otp,
            },
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

        _send_html_email(
            subject=subject,
            to_email=email,
            template_name="users/emails/login_success.html",
            context={
                "first_name": first_name,
                "email": email,
                "login_time": data.get("login_time", ""),
                "login_ip": data.get("login_ip", ""),
                "login_location": data.get("login_location", ""),
                "device_name": data.get("device_name") or data.get("device", ""),
                "browser": data.get("browser", ""),
                "os": data.get("os", ""),
                "login_method": data.get("login_method", ""),
                "session_id": data.get("session_id", ""),
                "dashboard_url": data.get("dashboard_url", "http://localhost:3000/dashboard/agent"),
                "secure_account_url": data.get("secure_account_url", "http://localhost:3000/help/contact"),
                "support_url": data.get("support_url", "http://localhost:3000/help"),
                "logo_url": data.get("logo_url", "https://ik.imagekit.io/pxc/DocuGyan/White-01.png"),
                "hero_image_url": data.get("hero_image_url", ""),
            },
        )
        return f"Success email sent to {email}"
    except Exception as e:
        logger.error(f"Failed to send success email to {email}: {str(e)}")
        raise e
