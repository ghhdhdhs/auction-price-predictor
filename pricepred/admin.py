from django.contrib import admin
from .models import Maker, Chassis, CarSale

@admin.register(Maker)
class MakerAdmin(admin.ModelAdmin):
    search_fields = ("name",)

@admin.register(Chassis)
class ChassisAdmin(admin.ModelAdmin):
    list_display = ("maker", "code")
    list_filter  = ("maker",)
    search_fields = ("code",)

@admin.register(CarSale)
class CarSaleAdmin(admin.ModelAdmin):
    autocomplete_fields = ("chassis_code",)
    list_display = ("auction_date","maker","chassis_code","year","mileage","color",
                    "auction_grade","interior_grade","start_price","final_price","auction_house")
    list_filter  = ("maker","chassis_code","auction_grade","interior_grade","auction_house","auction_date","color","year")
    search_fields = ("maker__name","chassis_code__code")
    date_hierarchy = "auction_date"
