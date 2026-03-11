import random
import string
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.exceptions import AuthenticationFailed

def generate_otp(length=6):
    """Generates a random numeric OTP."""
    return ''.join(random.choices(string.digits, k=length))

class CookieJWTAuthentication(JWTAuthentication):
    """
    Custom authentication class that extracts the JWT access token 
    from the HttpOnly cookie instead of the Authorization header.
    """
    def authenticate(self, request):
        header = self.get_header(request)
        if header is None:
            raw_token = request.COOKIES.get('access')
        else:
            raw_token = self.get_raw_token(header)

        if raw_token is None:
            return None

        try:
            validated_token = self.get_validated_token(raw_token)
            return self.get_user(validated_token), validated_token
        except Exception as e:
            raise AuthenticationFailed("Invalid or expired token")