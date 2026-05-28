"""Ticker registry and matching helpers for target EGX companies."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z\u0621-\u064a\s]")
_SPACE_RE = re.compile(r"\s+")
_ARABIC_NORMALIZATION_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ة": "ه",
        "ـ": "",
    }
)


def normalize_text(text: str) -> str:
    """Normalize Arabic/English text for alias matching."""

    if not text:
        return ""
    value = html.unescape(text)
    value = unicodedata.normalize("NFKC", value)
    value = _ARABIC_DIACRITICS_RE.sub("", value)
    value = value.translate(_ARABIC_NORMALIZATION_MAP)
    value = value.lower()
    value = _NON_ALNUM_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value).strip()
    return value


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


@dataclass(frozen=True)
class TargetCompany:
    """Canonical ticker metadata and name variants for one EGX company."""

    ticker: str
    site_code: str
    company_name_en: str
    company_name_ar: str
    primary_aliases: tuple[str, ...] = field(default_factory=tuple)
    secondary_aliases: tuple[str, ...] = field(default_factory=tuple)

    def all_aliases(self) -> List[str]:
        """Return every alias variant for matching and metadata."""

        return _dedupe(
            [
                self.ticker,
                self.site_code,
                self.company_name_en,
                self.company_name_ar,
                *self.primary_aliases,
                *self.secondary_aliases,
            ]
        )

    def match_text(self, text: str) -> Dict[str, List[str]]:
        """Return primary and secondary alias hits for a text snippet."""

        normalized_text = f" {normalize_text(text)} "
        primary_hits = [
            alias
            for alias in self.primary_aliases
            if f" {normalize_text(alias)} " in normalized_text
        ]
        secondary_hits = [
            alias
            for alias in self.secondary_aliases
            if f" {normalize_text(alias)} " in normalized_text
        ]
        return {
            "primary": _dedupe(primary_hits),
            "secondary": _dedupe(secondary_hits),
        }


TARGET_COMPANIES: Dict[str, TargetCompany] = {
    "CIB": TargetCompany(
        ticker="CIB",
        site_code="COMI",
        company_name_en="Commercial International Bank - Egypt",
        company_name_ar="البنك التجاري الدولي - مصر",
        primary_aliases=(
            "CIB",
            "COMI",
            "COMI.CA",
            "Commercial International Bank",
            "Commercial International Bank - Egypt",
            "Commercial International Bank-Egypt",
            "البنك التجاري الدولي",
            "البنك التجاري الدولي مصر",
            "البنك التجاري الدولي - مصر",
            "سي أي بي",
            "سي اي بي",
        ),
        secondary_aliases=(
            "التجاري الدولي",
            "CIB Egypt",
        ),
    ),
    "TMG": TargetCompany(
        ticker="TMG",
        site_code="TMGH",
        company_name_en="TMG Holding",
        company_name_ar="مجموعة طلعت مصطفى القابضة",
        primary_aliases=(
            "TMG",
            "TMGH",
            "TMGH.CA",
            "TMG Holding",
            "T M G Holding",
            "Talaat Moustafa Group",
            "Talaat Moustafa Group Holding",
            "Talaat Moustafa Holding Group",
            "مجموعة طلعت مصطفى",
            "مجموعة طلعت مصطفى القابضة",
            "طلعت مصطفى",
        ),
        secondary_aliases=(
            "TMG Group",
        ),
    ),
    "OIH": TargetCompany(
        ticker="OIH",
        site_code="OIH",
        company_name_en="Orascom Investment Holding",
        company_name_ar="أوراسكوم للاستثمار القابضة",
        primary_aliases=(
            "OIH",
            "OIH.CA",
            "Orascom Investment Holding",
            "Orascom Investment",
            "أوراسكوم للاستثمار القابضة",
            "اوراسكوم للاستثمار القابضة",
            "أوراسكوم للاستثمار",
            "اوراسكوم للاستثمار",
        ),
        secondary_aliases=(
            "OIH Holding",
            # Note: OIH was formerly OTMT (Orascom Telecom Media & Technology,
            # 2010-2018). OTMT aliases are intentionally excluded because nearly
            # all OTMT-era articles predate the business-model pivot to
            # diversified investment, and post-2018 OTMT mentions in the
            # Mubasher feed are predominantly Telecom Egypt stories — a
            # completely different company.
        ),
    ),
    "ORASCOM": TargetCompany(
        ticker="ORASCOM",
        site_code="ORAS",
        company_name_en="Orascom Construction PLC",
        company_name_ar="أوراسكوم كونستراكشون بي ال سي",
        primary_aliases=(
            "ORASCOM",
            "ORAS",
            "Orascom Construction",
            "Orascom Construction PLC",
            "أوراسكوم كونستراكشون",
            "اوراسكوم كونستراكشون",
            "أوراسكوم كونستراكشون بي ال سي",
            "اوراسكوم كونستراكشون بي ال سي",
        ),
        secondary_aliases=(
            "Orascom",
            "أوراسكوم",
            "اوراسكوم",
        ),
    ),
    "BELTONE": TargetCompany(
        ticker="BELTONE",
        site_code="BTFH",
        company_name_en="Beltone Holding",
        company_name_ar="بلتون القابضة",
        primary_aliases=(
            "BELTONE",
            "BTFH",
            "Beltone",
            "Beltone Holding",
            "Beltone Financial Holding",
            "بلتون",
            "بلتون القابضة",
            "بلتون المالية القابضة",
        ),
        secondary_aliases=(
            "Beltone Financial",
        ),
    ),
    "ELSEWEDY": TargetCompany(
        ticker="ELSEWEDY",
        site_code="SWDY",
        company_name_en="Elsewedy Electric Co SAE",
        company_name_ar="السويدي اليكتريك ش م م",
        primary_aliases=(
            "ELSEWEDY",
            "SWDY",
            "Elsewedy",
            "Elsewedy Electric",
            "Elsewedy Electric Co SAE",
            "El Sewedy Electric",
            "السويدي",
            "السويدي اليكتريك",
            "السويدي إليكتريك",
            "السويدي اليكتريك ش م م",
        ),
        secondary_aliases=(
            "Elsewedy Electric Co.",
        ),
    ),
}

_TICKER_ALIASES: Dict[str, str] = {}
for canonical_ticker, company in TARGET_COMPANIES.items():
    for alias in company.all_aliases():
        _TICKER_ALIASES[normalize_text(alias)] = canonical_ticker


def resolve_ticker(value: object) -> Optional[str]:
    """Resolve a raw ticker or alias to the project's canonical ticker label."""

    if value is None:
        return None
    normalized = normalize_text(str(value))
    if not normalized:
        return None
    return _TICKER_ALIASES.get(normalized, str(value).strip().upper() or None)
