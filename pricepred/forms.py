from django import forms
from .models import CarSale, Maker, Chassis

class CarSaleForm(forms.ModelForm):
    maker = forms.ModelChoiceField(queryset=Maker.objects.all())
    chassis_code = forms.ModelChoiceField(queryset=Chassis.objects.none())

    class Meta:
        model = CarSale
        fields = ["auction_date","maker","chassis_code","year","mileage","color",
                  "auction_grade","interior_grade","exterior_grade","start_price","final_price","auction_house"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If maker pre-selected (POST/initial), limit chassis queryset
        maker = (self.data.get("maker") or self.initial.get("maker"))
        if maker:
            self.fields["chassis_code"].queryset = Chassis.objects.filter(maker_id=maker)
