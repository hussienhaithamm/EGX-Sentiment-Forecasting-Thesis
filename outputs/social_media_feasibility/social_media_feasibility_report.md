# Social Media Source Feasibility

This experiment evaluates whether social-media data can be added to the EGX sentiment thesis using only ethical and officially permitted access paths.

## Platform Decision

| Platform | Official free/ethical path | Feasibility for this thesis | Decision |
| --- | --- | --- | --- |
| Reddit | Possible through Reddit Data API for approved non-commercial use or Reddit's research access path. Research use must remain academic, avoid redistribution, and avoid training models on Reddit data without explicit consent. | Feasible only after approval and with strict data-handling limits. Reddit volume for EGX-specific discussion is likely sparse. | Keep as future-work or run only with approved credentials. |
| X / Twitter | Official public reads currently require pay-per-use credits; owned reads are cheaper but cover only the authenticated user's own data. | Not free for broad EGX search. Unofficial scraping would be inappropriate for the thesis. | Exclude from current experiment; document as paid/institutional future work. |

Official source notes used for this decision:

- Reddit Help states that research use is allowed only for academic non-commercial purposes, subject to restrictions on redistribution and model training, and points researchers to Reddit's research access path.
- X API documentation describes a pay-per-use model for public API access; owned reads are cheaper but limited to the authenticated user's own resources.

## Reddit Collection Status

- Status: `SKIPPED`
- Records: `0`
- Reason: `Reddit collection was not requested. Use --run-reddit only after obtaining approved official API credentials.`

## EGX Query Plan

Recommended query terms combine market-level and ticker/company-level identifiers in Arabic and English.

```text
EGX
EGX30
البورصة المصرية
البورصه المصريه
البورصة
CIB
COMI
البنك التجاري الدولي
TMG
TMGH
طلعت مصطفى
OIH
اوراسكوم للاستثمار
```

Recommended subreddits for a pilot are:

```text
PersonalFinanceEgypt
Egypt
EgyptianStockMarket
investing
stocks
```

## Modeling Design If Approved Data Is Available

1. Collect posts through the official Reddit API or approved research API only.
2. Store only minimal metadata needed for reproducibility; avoid redistributing raw user content in the thesis artifact.
3. Apply entity matching for CIB, TMG, OIH, EGX, and Arabic aliases.
4. Route Arabic/English text through the same language-aware sentiment layer used for news.
5. Aggregate social sentiment daily with volume, sentiment mean, confidence, and ticker relevance features.
6. Compare news-only, social-only, news+social, and ensemble models under the same chronological splits and leakage checks.

## Paper-Safe Conclusion

The current thesis should present social media as future work, not as a completed experiment, unless approved Reddit credentials are obtained and the official API collection is run. X should be excluded from the free implementation path because broad public-read access is paid under the current official API model.

The most EGX-relevant prior work found is AlKhatib et al. (2016), which studied Arabic tweets and Arab market indices including EGX30 using Granger causality and Pearson correlation. That paper supports the relevance of Arabic social-media sentiment for future EGX-oriented work, but it does not solve the current platform-access constraint.
