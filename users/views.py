import os
import logging
import re
from django.core.cache import cache
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from django.conf import settings
from django.utils import timezone

from google.oauth2 import id_token
from google.auth.transport.requests import Request

from core.authentication import generate_otp, CookieJWTAuthentication
from core.cache import delete_all_user_cache

from .models import CustomUser  
from .tasks import Otp_Verification, send_login_success_email

logger = logging.getLogger(__name__)

# -------------------------
# Constants / helpers
# -------------------------

ACCESS_COOKIE = "access"
REFRESH_COOKIE = "refresh"
ACCESS_MAX_AGE = 30 * 60                # 30 minutes
REFRESH_MAX_AGE = 7 * 24 * 60 * 60      # 7 days
OTP_TTL_SECONDS = 140

GOOGLE_CLIENT_ID = settings.GOOGLE_CLIENT_ID


def jwt_cookie_opts():
    return {
        "httponly": True,
        "secure": True,
        "samesite": "None",
    }

def set_auth_cookies(response: Response, access_token: str, refresh_token: str):
    opts = jwt_cookie_opts()
    response.set_cookie(ACCESS_COOKIE, access_token, max_age=ACCESS_MAX_AGE, **opts)
    response.set_cookie(REFRESH_COOKIE, refresh_token, max_age=REFRESH_MAX_AGE, **opts)

def clear_auth_cookies(response: Response):
    response.delete_cookie(ACCESS_COOKIE)
    response.delete_cookie(REFRESH_COOKIE)

def issue_tokens_for_user(user: CustomUser):
    refresh = RefreshToken.for_user(user)
    return str(refresh.access_token), str(refresh)


def get_client_ip(request):
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return request.META.get("HTTP_X_REAL_IP") or request.META.get("REMOTE_ADDR", "")


def parse_user_agent(user_agent: str):
    ua = (user_agent or "").lower()

    if "edg/" in ua:
        browser = "Edge"
    elif "opr/" in ua or "opera" in ua:
        browser = "Opera"
    elif "chrome/" in ua and "edg/" not in ua:
        browser = "Chrome"
    elif "safari/" in ua and "chrome/" not in ua:
        browser = "Safari"
    elif "firefox/" in ua:
        browser = "Firefox"
    elif "trident/" in ua or "msie" in ua:
        browser = "Internet Explorer"
    else:
        browser = "Unknown Browser"

    if "windows" in ua:
        os_name = "Windows"
    elif "mac os x" in ua or "macintosh" in ua:
        os_name = "macOS"
    elif "android" in ua:
        os_name = "Android"
    elif "iphone" in ua or "ipad" in ua or "ios" in ua:
        os_name = "iOS"
    elif "linux" in ua:
        os_name = "Linux"
    else:
        os_name = "Unknown OS"

    mobile_pattern = re.compile(r"mobile|iphone|ipad|android")
    device_name = "Mobile Device" if mobile_pattern.search(ua) else "Desktop Device"

    return {
        "browser": browser,
        "os": os_name,
        "device_name": device_name,
    }


def build_login_email_payload(request, user: CustomUser, login_method: str):
    user_agent = request.META.get("HTTP_USER_AGENT", "")
    ua_data = parse_user_agent(user_agent)
    login_time = timezone.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    return {
        "email": user.email,
        "first_name": getattr(user, "first_name", "") or "User",
        "login_time": login_time,
        "login_ip": get_client_ip(request),
        "login_location": "",  # Fill if you add GeoIP lookup later
        "device_name": ua_data["device_name"],
        "browser": ua_data["browser"],
        "os": ua_data["os"],
        "login_method": login_method,
        "session_id": "",
        "dashboard_url": "https://docugyan.com",
        "secure_account_url": "https://docugyan.com/security",
        "support_url": "https://docugyan.com/support",
    }

# -------------------------
# Views
# -------------------------

class OtpVerificationView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        # Expecting OTP as a string or int, but converting to int for cache comparison
        otp_val = request.data.get("otp")
        user_id = request.data.get("id")
        
        if not user_id or not otp_val:
            return Response({"error": "Key (id) and OTP are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user_id = int(user_id)
            otp = int(otp_val)
        except ValueError:
            return Response({"error": "Invalid format for ID or OTP."}, status=status.HTTP_400_BAD_REQUEST)

        key = f"otp_{user_id}"
        cached_otp = cache.get(key)
        
        # Make sure cached_otp exists and matches (convert to int just in case)
        if cached_otp is None or int(cached_otp) != otp:
            return Response({"error": "Invalid key or OTP."}, status=status.HTTP_401_UNAUTHORIZED)

        user = CustomUser.objects.filter(id=user_id).first()
        if not user:
            return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)

        if not user.is_active:
            user.is_active = True
            user.save(update_fields=['is_active'])

        try:
            access_token, refresh_token = issue_tokens_for_user(user)
        except Exception as e:
            logger.error("Token generation failed for %s: %s", user.email, str(e))
            return Response({"error": "Failed to generate tokens."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        cache.delete(key)

        response_data = {
            "message": "OTP verified successfully.",
            "user_uuid": getattr(user, 'user_uuid', str(user.id))
        }
        response = Response(response_data, status=status.HTTP_200_OK)
        set_auth_cookies(response, access_token, refresh_token)

        try:
            send_login_success_email.delay(
                build_login_email_payload(request, user, login_method="OTP")
            )
        except Exception as e:
            logger.warning("Email send failed for %s: %s", user.email, str(e))

        return response


class Login_SignUpView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")
        if not email:
            return Response({"error": "Email is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Safely fetch or create. If an unverified user exists, reuse them to keep the ID stable.
            user, created = CustomUser.objects.get_or_create(
                email=email,
                defaults={"is_active": False},
            )
            status_message = "New User" if created else "Existing User"

            otp = generate_otp()
            key = f"otp_{user.id}"
            cache.set(key, otp, timeout=OTP_TTL_SECONDS)

            try:
                Otp_Verification.delay({"otp": otp, "email": user.email})
            except Exception as task_error:
                logger.warning("OTP async task enqueue failed for %s: %s", email, task_error)

            return Response({"key": key, "id": user.id, "status": status_message}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.exception("Login/Signup failed for email=%s: %s", email, e)
            return Response({"error": "Something went wrong. Try again later."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class LogoutView(APIView):
    authentication_classes = [CookieJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        refresh_token = request.COOKIES.get(REFRESH_COOKIE)
        if refresh_token:
            try:
                token = RefreshToken(refresh_token)
                token.blacklist()
            except Exception as e:
                logger.warning("Invalid refresh token during logout for user %s: %s", getattr(request.user, "email", "N/A"), str(e))
        
        response = Response({"message": "Logged out successfully."}, status=status.HTTP_200_OK)
        clear_auth_cookies(response)
        delete_all_user_cache(request.user)
        return response


class Google_Login_SignupView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        token_str = request.data.get("token")
        if not token_str:
            return Response({"error": "Token not provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            idinfo = id_token.verify_oauth2_token(token_str, Request(), GOOGLE_CLIENT_ID)
        except ValueError:
            return Response({"error": "Invalid token"}, status=status.HTTP_400_BAD_REQUEST)
            
        email = idinfo.get("email")
        if not email:
            return Response({"error": "Invalid token"}, status=status.HTTP_400_BAD_REQUEST)

        user_defaults = {
            "first_name": idinfo.get("given_name", ""),
            "last_name": idinfo.get("family_name", ""),
            "is_active": True, 
        }
        
        try:
            # This handles creating new users OR updating unverified users cleanly
            user, created = CustomUser.objects.update_or_create(
                email=email, defaults=user_defaults
            )
            status_message = "New User" if created else "Existing User"
        except Exception as e:
            logger.exception("User fetch/create failed for %s: %s", email, e)
            return Response({"error": "User creation failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            access_token, refresh_token = issue_tokens_for_user(user)
        except Exception as e:
            logger.error("JWT generation failed for %s: %s", email, e)
            return Response({"error": "Failed to generate tokens"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        response_data = {
            "message": "Login Successful",
            "status": status_message,
            "user_uuid": getattr(user, 'user_uuid', str(user.id))
        }
        response = Response(response_data, status=status.HTTP_200_OK)
        set_auth_cookies(response, access_token, refresh_token)

        try:
            send_login_success_email.delay(
                build_login_email_payload(request, user, login_method="Google OAuth")
            )
        except Exception as e:
            logger.warning("Async email enqueue failed for %s: %s", email, e)

        return response


class ResendOtpView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        user_id = request.data.get("id")
        if not user_id:
            return Response({"error": "ID is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Clear the old OTP key if the frontend happened to send it
        if key := request.data.get("key"):
            cache.delete(key)
            
        try:
            user = CustomUser.objects.filter(id=user_id, is_active=False).first()
            if not user:
                return Response({"error": "User not found or is already active."}, status=status.HTTP_404_NOT_FOUND)

            otp = generate_otp()
            new_key = f"otp_{user.id}"
            cache.set(new_key, otp, timeout=OTP_TTL_SECONDS)

            try:
                Otp_Verification.delay({"otp": otp, "email": user.email})
            except Exception as task_error:
                logger.warning("OTP async task enqueue failed for %s: %s", user.email, task_error)

            return Response({"key": new_key, "id": user.id}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.exception("Resend OTP failed for id=%s: %s", user_id, e)
            return Response({"error": "Something went wrong."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
