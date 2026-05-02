import os
import logging
import requests
from langchain_core.tools import tool

log = logging.getLogger("agent.apify")

APIFY_BASE = "https://api.apify.com/v2"
ACTOR_ID   = "maxcopell~zillow-scraper"
WAIT_SECS  = 25


@tool
def apify_get_property_images(address: str, city: str, state: str) -> dict:
    """Fetch property photos and floor plan URL using Apify's Zillow scraper."""
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        log.warning("[APIFY] ✗ APIFY_API_TOKEN not set — skipping images")
        return {"images": [], "floor_plan_url": None}

    zillow_url = _build_zillow_url(address, city, state)
    log.info(f"[APIFY] → Starting actor run for: {address}, {city} {state}")
    log.info(f"[APIFY]   Zillow URL: {zillow_url}")

    try:
        run_id = _start_actor_run(zillow_url, token)
        if not run_id:
            log.warning(f"[APIFY] ✗ Actor run failed to start for {address}")
            return {"images": [], "floor_plan_url": None}

        log.info(f"[APIFY]   Run ID: {run_id} — fetching results...")
        items = _get_dataset_items(run_id, token)

        if not items:
            log.warning(f"[APIFY] ✗ No results returned for {address}")
            return {"images": [], "floor_plan_url": None}

        result = _extract_media(items)
        log.info(f"[APIFY] ✓ {len(result['images'])} images found  floor_plan={'yes' if result['floor_plan_url'] else 'no'}")
        return result

    except Exception as e:
        log.error(f"[APIFY] ✗ Exception for {address}: {e}")
        return {"images": [], "floor_plan_url": None}


def _build_zillow_url(address: str, city: str, state: str) -> str:
    slug = f"{address} {city} {state}".lower()
    for ch in [",", ".", "#"]:
        slug = slug.replace(ch, "")
    return f"https://www.zillow.com/homes/{slug.replace(' ', '-')}/"


def _start_actor_run(start_url: str, token: str) -> str | None:
    resp = requests.post(
        f"{APIFY_BASE}/acts/{ACTOR_ID}/runs",
        params={"token": token, "waitForFinish": WAIT_SECS},
        json={"searchUrls": [{"url": start_url}], "maxItems": 3},
        timeout=WAIT_SECS + 10,
    )
    if not resp.ok:
        log.error(f"[APIFY] ✗ Actor start HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    return resp.json().get("data", {}).get("id")


def _get_dataset_items(run_id: str, token: str) -> list:
    resp = requests.get(
        f"{APIFY_BASE}/actor-runs/{run_id}/dataset/items",
        params={"token": token},
        timeout=15,
    )
    if not resp.ok:
        log.error(f"[APIFY] ✗ Dataset fetch HTTP {resp.status_code}")
        return []
    data = resp.json()
    return data if isinstance(data, list) else []


def _extract_media(items: list) -> dict:
    if not items:
        return {"images": [], "floor_plan_url": None}
    first = items[0]
    raw_photos = first.get("photos", []) or first.get("images", [])
    images: list[str] = []
    for p in raw_photos[:6]:
        if isinstance(p, str):
            images.append(p)
        elif isinstance(p, dict):
            url = p.get("url") or p.get("src") or p.get("href")
            if url:
                images.append(url)
    floor_plan: str | None = None
    fp = first.get("floorPlan") or first.get("floor_plan")
    if isinstance(fp, dict):
        floor_plan = fp.get("url") or fp.get("src")
    elif isinstance(fp, str):
        floor_plan = fp
    return {"images": images, "floor_plan_url": floor_plan}
