!pip install -q google-genai
from kaggle_secrets import UserSecretsClient
import os

user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GEMINI_API_KEY")

os.environ["GEMINI_API_KEY"] = api_key
print("Gemini key loaded!")
import os, json, re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import date

# ======================================================
# 1. GEMINI SETUP (SAFE – NO CRASH IF NO KEY)
# ======================================================

try:
    from google import genai
    _api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if _api_key:
        _genai_client = genai.Client(api_key=_api_key)
        GEMINI_AVAILABLE = True
    else:
        _genai_client = None
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False
    _genai_client = None


def today_str() -> str:
    return date.today().isoformat()


def parse_price(s: str) -> Optional[float]:
    s = s.strip()
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else None


# ======================================================
# 2. AGENT 1 – RECEIPT EXTRACTOR
# ======================================================

@dataclass
class Item:
    name: str
    qty: float
    price: float


class ReceiptExtractorAgent:
    def extract(self, receipt_text: str) -> Dict[str, Any]:
        lines = receipt_text.splitlines()

        store = "Unknown Store"
        for line in lines:
            if line.strip():
                store = line.strip()
                break

        rec_date = today_str()
        for line in lines:
            m = re.search(r"\d{4}-\d{2}-\d{2}", line)
            if m:
                rec_date = m.group(0)
                break

        items: List[Item] = []
        ignore_keywords = ["total", "subtotal", "tax", "visa", "mastercard"]

        for line in lines:
            raw = line.strip()
            if not raw:
                continue
            lower = raw.lower()
            if any(k in lower for k in ignore_keywords):
                continue

            if "-" in raw:
                left, right = raw.rsplit("-", 1)
                price = parse_price(right)
                if price is not None:
                    items.append(Item(name=left.strip(), qty=1.0, price=price))
                    continue

            tokens = raw.split()
            if len(tokens) >= 2:
                price = parse_price(tokens[-1])
                if price is not None:
                    name = " ".join(tokens[:-1])
                    items.append(Item(name=name, qty=1.0, price=price))

        return {
            "store": store,
            "date": rec_date,
            "items": [asdict(i) for i in items],
        }


# ======================================================
# 3. AGENT 2 – CARBON CALCULATOR
# ======================================================

class CarbonCalculatorAgent:
    CATEGORY_FACTORS = {
        "red_meat": 20.0,
        "chicken": 6.0,
        "fish": 5.0,
        "dairy": 3.0,
        "cheese": 8.0,
        "eggs": 3.0,
        "vegetables": 0.5,
        "fruits": 0.5,
        "grains": 1.0,
        "snacks": 2.0,
        "electronics": 50.0,
        "clothing": 15.0,
        "household": 5.0,
        "other": 2.0,
    }

    def _categorize(self, name: str) -> str:
        n = name.lower()
        if any(k in n for k in ["beef", "steak", "lamb", "mutton", "pork", "burger"]):
            return "red_meat"
        if any(k in n for k in ["chicken", "turkey"]):
            return "chicken"
        if any(k in n for k in ["fish", "salmon", "tuna", "shrimp"]):
            return "fish"
        if any(k in n for k in ["milk", "yogurt", "cream"]):
            return "dairy"
        if "cheese" in n:
            return "cheese"
        if "egg " in n:
            return "eggs"
        if any(k in n for k in ["apple", "banana", "orange", "mango", "grape"]):
            return "fruits"
        if any(k in n for k in ["lettuce", "spinach", "carrot", "broccoli"]):
            return "vegetables"
        if any(k in n for k in ["rice", "pasta", "bread", "flour", "oats", "cereal"]):
            return "grains"
        if any(k in n for k in ["chips", "chocolate", "cookie", "snack"]):
            return "snacks"
        if any(k in n for k in ["phone", "laptop", "charger", "monitor", "tv"]):
            return "electronics"
        if any(k in n for k in ["shirt", "jeans", "dress", "jacket", "t-shirt"]):
            return "clothing"
        if any(k in n for k in ["detergent", "cleaner", "soap", "shampoo", "tissue"]):
            return "household"
        return "other"

    def compute(self, parsed_receipt: Dict[str, Any]) -> Dict[str, Any]:
        enriched = []
        total_co2 = 0.0

        for it in parsed_receipt.get("items", []):
            name = it["name"]
            qty = it.get("qty", 1.0)
            category = self._categorize(name)
            factor = self.CATEGORY_FACTORS.get(category, self.CATEGORY_FACTORS["other"])
            co2 = factor * qty
            total_co2 += co2
            enriched.append({
                **it,
                "category": category,
                "co2_kg": co2,
            })

        return {
            "store": parsed_receipt["store"],
            "date": parsed_receipt["date"],
            "items": enriched,
            "total_co2_kg": total_co2,
        }


# ======================================================
# 4. AGENT 3 – PROGRESS TRACKER
# ======================================================

class ProgressTrackerAgent:
    def __init__(self, path: str = "progress_history.json"):
        self.path = path
        self._data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def update(self, user_id: str, date_str: str, total_co2: float) -> Dict[str, Any]:
        history = self._data.get(user_id, [])
        history.append({"date": date_str, "total_co2_kg": total_co2})
        history = sorted(history, key=lambda x: x["date"])
        self._data[user_id] = history
        self._save()

        num = len(history)
        total = sum(h["total_co2_kg"] for h in history)
        avg = total / num
        score = max(0, min(100, int(100 - 2 * avg)))
        return {"history": history, "avg_daily": avg, "sustainability_score": score}


# ======================================================
# 5. AGENT 4 – RECOMMENDATION (GEMINI OR GENERIC)
# ======================================================

class RecommendationAgent:
    SYSTEM_PROMPT = (
        "You are a friendly, practical sustainability coach. "
        "Given a shopping receipt analysis, explain in simple language:\n"
        "1) Which categories or items have the highest carbon impact,\n"
        "2) 3–5 small, realistic changes the user could try next month,\n"
        "3) Keep the tone supportive and non-judgmental.\n"
        "Use short bullet points where helpful."
    )

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name

    def _build_prompt(self, carbon_result: Dict[str, Any], progress: Dict[str, Any]) -> str:
        lines = []
        lines.append(f"Total emissions this receipt: {carbon_result['total_co2_kg']:.1f} kg CO2e.")
        lines.append("Items:")
        for item in carbon_result["items"]:
            lines.append(
                f"- {item['name']} (category={item['category']}, co2={item['co2_kg']:.1f} kg)"
            )
        lines.append("")
        lines.append(
            f"User history: {len(progress['history'])} receipt(s) logged, "
            f"avg daily footprint from receipts ~{progress['avg_daily']:.1f} kg CO2e, "
            f"current sustainability_score={progress['sustainability_score']}/100."
        )
        return "\n".join(lines)

    def recommend(self, carbon_result: Dict[str, Any], progress: Dict[str, Any]) -> str:
        if not GEMINI_AVAILABLE or _genai_client is None:
            tips = [
                "Try swapping one red-meat meal per week for a vegetarian or plant-based option.",
                "Combine errands into a single trip to reduce transport-related emissions.",
                "Delay non-essential electronics or fashion purchases when possible.",
                "Look for local or seasonal produce to lower food-related emissions.",
            ]
            return "Gemini not configured, showing generic tips:\n" + "\n".join(f"- {t}" for t in tips)

        user_context = self._build_prompt(carbon_result, progress)
        contents = [
            {"role": "user", "parts": [{"text": self.SYSTEM_PROMPT}]},
            {"role": "user", "parts": [{"text": user_context}]},
        ]
        resp = _genai_client.models.generate_content(
            model=self.model_name,
            contents=contents,
        )
        return resp.text


# ======================================================
# 6. ORCHESTRATOR + HELPER FUNCTIONS
# ======================================================

class SustainableLivingCoach:
    def __init__(self, history_path: str = "progress_history.json"):
        self.extractor = ReceiptExtractorAgent()
        self.carbon = CarbonCalculatorAgent()
        self.progress = ProgressTrackerAgent(history_path)
        self.recommender = RecommendationAgent()

    def run_on_text(self, receipt_text: str, user_id: str = "demo_user") -> Dict[str, Any]:
        parsed = self.extractor.extract(receipt_text)
        carbon = self.carbon.compute(parsed)
        progress = self.progress.update(
            user_id=user_id,
            date_str=carbon["date"],
            total_co2=carbon["total_co2_kg"],
        )
        recs = self.recommender.recommend(carbon, progress)
        return {
            "parsed_receipt": parsed,
            "carbon_result": carbon,
            "progress_summary": progress,
            "recommendations": recs,  # this is a STRING
        }


# global coach instance for notebook use
coach = SustainableLivingCoach()

def run_receipt_from_text(receipt_text: str, user_id: str = "demo_user"):
    """Paste text directly in notebook."""
    result = coach.run_on_text(receipt_text, user_id=user_id)

    print("=== Parsed Receipt ===")
    print(json.dumps(result["parsed_receipt"], indent=2))

    print("\n=== Carbon Result ===")
    print(json.dumps(result["carbon_result"], indent=2))

    print("\n=== Progress Summary ===")
    print(json.dumps(result["progress_summary"], indent=2))

    print("\n=== Recommendations (LLM or generic) ===")
    print(result["recommendations"])

    return result


def run_receipt_from_file(path: str, user_id: str = "demo_user"):
    """Read a .txt file and run the full pipeline."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r") as f:
        receipt_text = f.read()

    return run_receipt_from_text(receipt_text, user_id=user_id)


def run_with_user_input_notebook():
    """Interactive menu like your old script: paste OR file path."""
    print("Sustainable Living Coach Agent")
    print("------------------------------")
    print("1) Paste receipt text")
    print("2) Use uploaded .txt file path (/kaggle/input/...)")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        print("\nPaste your receipt text. End input with an empty line:")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        text = "\n".join(lines)
        return run_receipt_from_text(text)

    elif choice == "2":
        print("\nGo to the right 'Data' tab and copy the full path, e.g.")
        print("  /kaggle/input/my-receipt-dataset/receipt1.txt\n")
        path = input("Enter full file path: ").strip()
        return run_receipt_from_file(path)

    else:
        print("Invalid choice.")
        return None

print("Coach ready. GEMINI_AVAILABLE =", GEMINI_AVAILABLE)
sample = """
Walmart Supercenter
2025-11-20
Beef steak - 12.50
Milk 2L - 3.20
Apples x3 4.10
TOTAL 19.80
"""

result = run_receipt_from_text(sample, user_id="student_01")
from textwrap import shorten

# global coach instance
coach = SustainableLivingCoach()

def _bar(value, max_value=40, width=20):
    """Simple text progress bar for scores / emissions."""
    ratio = min(max(value / max_value, 0), 1)
    filled = int(ratio * width)
    return "█" * filled + "·" * (width - filled)


def run_receipt_from_text(receipt_text: str, user_id: str = "demo_user"):
    """Pretty console dashboard for a receipt."""
    result = coach.run_on_text(receipt_text, user_id=user_id)
    parsed = result["parsed_receipt"]
    carbon = result["carbon_result"]
    prog   = result["progress_summary"]
    recs   = result["recommendations"]

    store = parsed["store"]
    date  = parsed["date"]
    items = carbon["items"]
    total_co2 = carbon["total_co2_kg"]
    score = prog["sustainability_score"]

    # ---------- HEADER ----------
    print("\n🌱  Sustainable Living Coach")
    print("──────────────────────────────")
    print(f"🛒  Store      : {store}")
    print(f"📅  Date       : {date}")
    print(f"🧾  Items      : {len(items)}")
    print(f"🌍  Total CO₂  : {total_co2:.1f} kg")
    print(f"💚  Score      : {score}/100  {_bar(score, max_value=100)}")

    # ---------- PER-ITEM TABLE ----------
    print("\n📦  Item Breakdown")
    print("──────────────────────────────")
    print(f"{'Item':40}  {'Cat':10}  {'CO₂ (kg)':7}")
    print("-" * 65)
    for item in items:
        name = shorten(item["name"], width=40, placeholder="…")
        cat  = item["category"]
        co2  = item["co2_kg"]
        print(f"{name:40}  {cat:10}  {co2:7.1f}")

    # ---------- PROGRESS SUMMARY ----------
    print("\n📈  Your Progress So Far")
    print("──────────────────────────────")
    print(f"Days logged   : {len(prog['history'])}")
    print(f"Avg daily CO₂ : {prog['avg_daily']:.1f} kg")
    print(f"Score trend   : {_bar(score, max_value=100)}")

    # ---------- RECOMMENDATIONS ----------
    print("\n💡  Coach Suggestions")
    print("──────────────────────────────")
    print(recs)   # already nicely formatted by Gemini or our generic tips

    return result
def _parse_line_to_item(self, line: str) -> Optional[Dict[str, Any]]:
    """
    Improved parser:
    - Accepts lines WITHOUT price
    - Accepts lines WITH or WITHOUT dash
    - Accepts items like "Fresh Roma Tomato, Each"
    """
    original = line.strip()
    if not original:
        return None

    lower = original.lower()
    ignore_keywords = ["total", "subtotal", "tax", "visa", "mastercard"]
    if any(k in lower for k in ignore_keywords):
        return None

    # Case 1: Look for price at end
    tokens = original.split()
    if len(tokens) >= 2:
        price = parse_price(tokens[-1])
        if price is not None:
            # with price
            name = " ".join(tokens[:-1])
            return {"name": name, "qty": 1, "price": price}

    # Case 2: Look for dash pattern
    if "-" in original:
        left, right = original.rsplit("-", 1)
        price = parse_price(right)
        if price is not None:
            return {"name": left.strip(), "qty": 1, "price": price}

    # Case 3: NO PRICE (new behavior)
    # treat entire line as an item with automatic category assignment
    return {"name": original, "qty": 1, "price": None}


def run_receipt_from_file(path: str, user_id: str = "demo_user"):
    """Read a .txt file and run the full pretty dashboard."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r") as f:
        receipt_text = f.read()

    return run_receipt_from_text(receipt_text, user_id=user_id)


def run_with_user_input_notebook():
    """Interactive menu (paste OR file path) with pretty output."""
    print("Sustainable Living Coach Agent")
    print("------------------------------")
    print("1) Paste receipt text")
    print("2) Use uploaded .txt file path (/kaggle/input/...)")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        print("\nPaste your receipt text. End input with an empty line:")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        text = "\n".join(lines)
        return run_receipt_from_text(text)

    elif choice == "2":
        print("\nIn the right 'Data' tab, copy the full file path, e.g.:")
        print("  /kaggle/input/my-receipt-dataset/receipt1.txt\n")
        path = input("Enter full file path: ").strip()
        return run_receipt_from_file(path)

    else:
        print("Invalid choice.")
        return None
run_with_user_input_notebook()
