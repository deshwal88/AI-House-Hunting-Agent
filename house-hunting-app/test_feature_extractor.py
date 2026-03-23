"""
test_feature_extractor.py
--------------------------
Run this to verify that feature extraction works correctly across
a range of realistic user queries.

Usage:
    python test_feature_extractor.py --api-key YOUR_GEMINI_KEY

    # Or set env var and skip the flag:
    export GEMINI_API_KEY=your_key
    python test_feature_extractor.py
"""

import sys
import json
import time

from feature_extractor import extract_features, format_requirements
import streamlit as st
api_key = st.secrets["GEMINI_API_KEY"]

# ── Test cases ───────────────────────────────────────────────────────────────
TEST_QUERIES = [
    {
        "query": "2 bedroom apartment under $2400 near Rutgers with parking and grocery stores nearby",
        "expect": {
            "bedrooms": 2,
            "max_budget": 2400,
            "parking_required": True,
        }
    },
    {
        "query": "1 bed studio in New Brunswick under $1800, commute under 20 min, pet friendly",
        "expect": {
            "bedrooms": 1,
            "max_budget": 1800,
            "pet_friendly": True,
            "max_commute_minutes": 20,
        }
    },
    {
        "query": "Looking for a 3 bed house between $2000 and $3000/month near NYC with gym and laundry in unit",
        "expect": {
            "bedrooms": 3,
            "min_budget": 2000,
            "max_budget": 3000,
        }
    },
    {
        "query": "Affordable studio near downtown Jersey City, need parking, no specific budget",
        "expect": {
            "bedrooms": 0,
            "parking_required": True,
            "city": "Jersey City",
        }
    },
    {
        "query": "2 bed condo in Hoboken, $3500 max, near PATH train, rooftop gym preferred",
        "expect": {
            "bedrooms": 2,
            "max_budget": 3500,
            "property_type": "Condo",
            "city": "Hoboken",
        }
    },
    {
        "query": "3 bed house in Austin TX, at least 1200 sqft, under $3000, built after 2005, pet friendly",
        "expect": {
            "bedrooms": 3,
            "max_budget": 3000,
            "city": "Austin",
            "state": "TX",
            "square_footage": 1200,
            "square_footage_operator": "min",
            "year_built": 2005,
            "year_built_operator": "min",
            "pet_friendly": True,
        }
    },
    {
        "query": "apartment near 220 Lincoln Avenue Newark NJ 07102, no more than 900 sqft, under $1500",
        "expect": {
            "city": "Newark",
            "state": "NJ",
            "zip_code": "07102",
            "max_budget": 1500,
            "square_footage": 900,
            "square_footage_operator": "max",
        }
    },
]

# ── Runner ────────────────────────────────────────────────────────────────────
PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

def check(req, expect: dict) -> list[str]:
    """Return list of failure messages (empty = all passed)."""
    failures = []
    req_dict = req.model_dump()
    soft_match_fields = {"location", "city", "state", "commute_destination"}
    for field, expected_val in expect.items():
        actual = req_dict.get(field)
        if field in soft_match_fields:
            # Soft check — just verify expected string appears in actual
            if not actual or expected_val.lower() not in actual.lower():
                failures.append(f"  {field}: expected to contain '{expected_val}', got '{actual}'")
        elif actual != expected_val:
            failures.append(f"  {field}: expected {expected_val!r}, got {actual!r}")
    return failures

START_FROM = 1  # Set to 1 to run all, or higher to skip already-passed cases

def run_tests():
    print("=" * 60)
    print("  Feature Extractor — Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, tc in enumerate(TEST_QUERIES, 1):
        if i < START_FROM:
            print(f"\n[{i}/{len(TEST_QUERIES)}] SKIPPED")
            print("-" * 60)
            continue
        query = tc["query"]
        expect = tc["expect"]

        print(f"\n[{i}/{len(TEST_QUERIES)}] {query[:70]}...")
        try:
            req = extract_features(query, api_key)
            failures = check(req, expect)

            if failures:
                print(f"{FAIL} FAIL")
                for f in failures:
                    print(f)
                failed += 1
            else:
                print(f"{PASS} PASS")
                passed += 1

            # Always print what was extracted
            print(format_requirements(req))
            print()
            print("  Raw JSON:")
            print("  " + json.dumps(req.model_dump(), indent=2).replace("\n", "\n  "))

        except Exception as e:
            print(f"{FAIL} ERROR — {e}")
            failed += 1

        print("-" * 60)
        if i < len(TEST_QUERIES):
            time.sleep(13)  # stay under 5 req/min free tier limit

    print(f"\n📊 Results: {passed}/{len(TEST_QUERIES)} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)