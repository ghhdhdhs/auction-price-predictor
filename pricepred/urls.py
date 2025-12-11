"""
URL configuration for pricepred project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from .views import (
    add_car_sale,
    chassis_options,
    api_predict,
    dashboard,
    api_history,
)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("add-sale", add_car_sale, name="add_car_sale"),
    path("chassis-options", chassis_options, name="chassis_options"),
    path("api/predict", api_predict, name="api_predict"),
    path("api/history", api_history, name="api_history"),
    path("", dashboard, name="dashboard"),  # root = your main page
]
