"""Single source of truth for EGX target ticker metadata.

Internal canonical labels:  CIB, TMG, OIH
Mubasher site codes:        COMI, TMGH, OIH   (used in listing URLs)
EGX exchange codes:         COMI, TMGH, OIH

Notes:
  - OIH (Orascom Investment Holding) was formerly registered as OTMT
    (Orascom Telecom Media & Technology).  Historical coverage before the
    2018 name change may use OTMT aliases, which are included as secondary
    aliases so the confidence filter can score them correctly.
  - The Arabic sentiment pipeline uses AraBERT; the English pipeline uses
    FinBERT.  Both are invoked by the model_route field in the pipeline.
"""

from __future__ import annotations

TICKERS: dict = {
    "CIB": {
        "exchange_code": "COMI",
        "mubasher_site_code": "COMI",
        "canonical_label": "CIB",
        "company_name_en": "Commercial International Bank - Egypt",
        "company_name_ar": "البنك التجاري الدولي - مصر",
        "short_name_en": "CIB",
        "short_name_ar": "CIB",
        "sector": "Banking",
        "search_terms_en": [
            "CIB",
            "COMI",
            "Commercial International Bank",
            "Commercial International Bank - Egypt",
            "Commercial International Bank-Egypt",
            "CIB Egypt",
        ],
        "search_terms_ar": [
            "البنك التجاري الدولي",
            "البنك التجاري الدولي مصر",
            "البنك التجاري الدولي - مصر",
            "سي أي بي",
            "سي اي بي",
            "التجاري الدولي",
            "COMI",
        ],
        "mubasher_news_url_ar": "https://www.mubasher.info/markets/EGX/stocks/COMI/news/",
        "mubasher_news_url_en": "https://english.mubasher.info/markets/EGX/stocks/COMI/news/",
        "price_file": "data/Commercial Int Bank Stock Price History.csv",
        "notes": "Largest private bank in Egypt by assets.",
    },
    "TMG": {
        "exchange_code": "TMGH",
        "mubasher_site_code": "TMGH",
        "canonical_label": "TMG",
        "company_name_en": "Talaat Moustafa Group Holding",
        "company_name_ar": "مجموعة طلعت مصطفى القابضة",
        "short_name_en": "TMG",
        "short_name_ar": "طلعت مصطفى",
        "sector": "Real Estate",
        "search_terms_en": [
            "TMG",
            "TMGH",
            "TMG Holding",
            "Talaat Moustafa",
            "Talaat Moustafa Group",
            "Talaat Moustafa Group Holding",
            "Talaat Moustafa Holding Group",
            "T M G Holding",
            "TMG Group",
        ],
        "search_terms_ar": [
            "طلعت مصطفى",
            "مجموعة طلعت مصطفى",
            "مجموعة طلعت مصطفى القابضة",
            "TMGH",
        ],
        "mubasher_news_url_ar": "https://www.mubasher.info/markets/EGX/stocks/TMGH/news/",
        "mubasher_news_url_en": "https://english.mubasher.info/markets/EGX/stocks/TMGH/news/",
        "price_file": "data/TMG Holding Stock Price History.csv",
        "notes": "Largest real estate developer in Egypt by land bank.",
    },
    "OIH": {
        "exchange_code": "OIH",
        "mubasher_site_code": "OIH",
        "canonical_label": "OIH",
        "company_name_en": "Orascom Investment Holding",
        "company_name_ar": "أوراسكوم للاستثمار القابضة",
        "short_name_en": "OIH",
        "short_name_ar": "أوراسكوم",
        "sector": "Diversified Investment",
        "search_terms_en": [
            "OIH",
            "Orascom Investment Holding",
            "Orascom Investment",
            "OIH Holding",
        ],
        "search_terms_ar": [
            "أوراسكوم للاستثمار القابضة",
            "اوراسكوم للاستثمار القابضة",
            "أوراسكوم للاستثمار",
            "اوراسكوم للاستثمار",
        ],
        "mubasher_news_url_ar": "https://www.mubasher.info/markets/EGX/stocks/OIH/news/",
        "mubasher_news_url_en": "https://english.mubasher.info/markets/EGX/stocks/OIH/news/",
        "price_file": "data/Orascom Investment Holding.csv",
        "notes": (
            "Formerly registered as OTMT (Orascom Telecom Media & Technology). "
            "Name changed to Orascom Investment Holding in 2018 when the company "
            "pivoted from telecom operations to diversified investment. "
            "Historical articles from 2010-2018 may use the OTMT name."
        ),
    },
}

# Convenience: all three canonical labels in a stable order
TICKER_LABELS: list[str] = ["CIB", "TMG", "OIH"]
