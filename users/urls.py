from django.urls import path
from .views import (
    Login_SignUpView, 
    OtpVerificationView, 
    LogoutView, 
    Google_Login_SignupView, 
    ResendOtpView
)

urlpatterns = [
    path('Login_SignUp/', Login_SignUpView.as_view(), name='login'),
    path('otp-verify/', OtpVerificationView.as_view(), name='otp-verify'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('google/', Google_Login_SignupView.as_view(), name='Google Login/SignUp'),
    path('resend-otp/', ResendOtpView.as_view(), name='resend-otp'),
]