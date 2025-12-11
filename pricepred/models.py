"""
Database models for auction vehicle records used by the price prediction system.

This module defines three related entities:

1. Maker
   ------------------------------------------------
   Represents a vehicle manufacturer (e.g. Mercedes-Benz, BMW). Each Maker
   may have multiple associated chassis codes.

2. Chassis
   ------------------------------------------------
   Identifies a manufacturer-specific platform / chassis code, optionally
   with a variant string. The (maker, code, variant) combination is unique.
   Examples:
       Maker = Mercedes-Benz
       Code  = S213
       Variant = E400 4MATIC

3. CarSale
   ------------------------------------------------
   Represents a single auction transaction. Stores:
       • Auction details (house, date)
       • Vehicle details (maker, chassis, year, mileage, colour)
       • JP auction grading (overall, interior, exterior)
       • Price information (start price, final price)

   These records form the authoritative dataset used for ML training.

------------------------------------------------------------------------------  
Design Notes
------------------------------------------------------------------------------
• AuctionGrade and LetterGrade enums follow Japanese auction standards.
• Color choices are normalised to a compact, modelling-friendly set.
• Foreign keys use PROTECT to avoid accidental data loss.
• __str__() methods are human-readable for admin and debugging.

This schema is intentionally minimal and stable, designed for repeatable
model training without schema drift.
"""

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator


class Maker(models.Model):
    """
    Vehicle manufacturer (e.g. Mercedes-Benz, BMW).

    This is kept as a separate model so that multiple chassis codes
    can be associated with the same maker.
    """
    name = models.CharField(max_length=50, unique=True)

    def __str__(self) -> str:
        return self.name


class Chassis(models.Model):
    """
    Chassis code / platform for a given maker.

    Examples:
        maker  = "Mercedes-Benz"
        code   = "S213"
        variant = "E400 4MATIC"

    The (maker, code, variant) combination is enforced as unique.
    """
    maker   = models.ForeignKey(
        Maker,
        on_delete=models.PROTECT,
        related_name="chassis_codes",
    )
    code    = models.CharField(max_length=50)
    variant = models.CharField(max_length=50, blank=True)

    class Meta:
        unique_together = ("maker", "code", "variant")

    def __str__(self) -> str:
        # .strip() handles the case where variant is empty
        return f"{self.maker.name} {self.code} {self.variant}".strip()


class CarSale(models.Model):
    """
    Single auction sale record for a car.

    Captures:
        - Auction metadata (house, date)
        - Vehicle identity (maker, chassis, year, mileage, colour)
        - Japanese-style auction grades (overall, interior, exterior)
        - Price information (start and final hammer price)

    These records form the core dataset used for model training.
    """

    # --- JP auction dropdowns / enums -----------------------------
    class AuctionGrade(models.TextChoices):
        """
        Overall auction grade (Japanese auction standard).

        Notes:
            S   : showroom / almost new
            6-1 : numeric condition grades
            RA  : minor repaired
            R   : repaired
        """
        S   = "S",   "S"     # very new / showroom
        G6  = "6",   "6"
        G5  = "5",   "5"
        G45 = "4.5", "4.5"
        G4  = "4",   "4"
        G35 = "3.5", "3.5"
        G3  = "3",   "3"
        G2  = "2",   "2"
        G1  = "1",   "1"
        RA  = "RA",  "RA"    # minor repaired
        R   = "R",   "R"     # repaired

    class LetterGrade(models.TextChoices):
        """
        Letter-based condition grades, typically used for interior/exterior.
        """
        A = "A", "A"
        B = "B", "B"
        C = "C", "C"
        D = "D", "D"
        E = "E", "E"

    class Color(models.TextChoices):
        """
        Normalised colour palette used for modelling.

        Free-text colours from the auction can be mapped into this set.
        """
        WHITE  = "White",  "White"
        BLACK  = "Black",  "Black"
        SILVER = "Silver", "Silver"
        GRAY   = "Gray",   "Gray"
        BLUE   = "Blue",   "Blue"
        RED    = "Red",    "Red"
        OTHER  = "Other",  "Other"

    # --- Auction info ---------------------------------------------
    auction_house = models.CharField(
        max_length=100,
        blank=True,
        help_text="Name of the auction house (e.g. USS Tokyo).",
    )
    auction_date = models.DateField(
        help_text="Date of the auction.",
    )

    # --- Dynamic maker / chassis ---------------------------------
    maker = models.ForeignKey(
        Maker,
        on_delete=models.PROTECT,
        help_text="Vehicle manufacturer.",
    )
    chassis_code = models.ForeignKey(
        Chassis,
        on_delete=models.PROTECT,
        help_text="Chassis / platform code for this vehicle.",
    )

    # --- Core vehicle details ------------------------------------
    year = models.IntegerField(
        validators=[MinValueValidator(1980), MaxValueValidator(2100)],
        help_text="Model year (Gregorian calendar).",
    )
    mileage = models.IntegerField(
        help_text="Odometer reading in kilometres.",
        validators=[MinValueValidator(0)],
    )

    # --- Auction grades & colour (JP-specific) -------------------
    color = models.CharField(
        max_length=30,
        choices=Color.choices,
        help_text="Normalised exterior colour.",
    )
    auction_grade = models.CharField(
        max_length=5,
        choices=AuctionGrade.choices,
        help_text="Overall auction grade.",
    )
    interior_grade = models.CharField(
        max_length=2,
        choices=LetterGrade.choices,
        help_text="Interior condition grade.",
    )
    exterior_grade = models.CharField(
        max_length=2,
        choices=LetterGrade.choices,
        blank=True,
        help_text="Exterior condition grade (optional).",
    )

    # --- Prices ---------------------------------------------------
    start_price = models.IntegerField(
        blank=True,
        null=True,
        help_text="Starting price in JPY (if available).",
        validators=[MinValueValidator(0)],
    )
    final_price = models.IntegerField(
        help_text="Final hammer price in JPY.",
        validators=[MinValueValidator(0)],
    )

    # --- Metadata -------------------------------------------------
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when this record was created.",
    )

    def __str__(self) -> str:
        return (
            f"{self.maker} {self.chassis_code.code} {self.year} "
            f"({self.mileage} km, {self.color}) - ¥{self.final_price}"
        )
