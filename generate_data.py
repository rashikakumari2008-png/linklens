"""
generate_data.py  -  Step 1 of LinkLens

Creates a FICTIONAL dataset of cyber-fraud reports with a hidden answer key.

Outputs (in the data/ folder):
  reports.csv       what an investigator would see (no answer key inside)
  ground_truth.csv  which reports truly belong to the same fraud gang (used ONLY for evaluation)

Everything is fake: phone numbers start with "00" (not valid in real life),
URLs end in ".example" (a reserved fake domain), and UPI handles end in "@fakebank".

Run:  python generate_data.py
Same seed -> same data every time (so your results are repeatable).
"""
import argparse
import csv
import os
import random
from datetime import date, timedelta
from itertools import combinations

# ----------------------------------------------------------------------------
# Building blocks (all fictional names)
# ----------------------------------------------------------------------------
CITIES = ["Delhi", "Mumbai", "Pune", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
          "Jaipur", "Lucknow", "Ahmedabad", "Chandigarh", "Bhopal", "Patna", "Indore",
          "Nagpur", "Kochi"]
BANKS = ["Sunrise Bank", "Metro Co-op Bank", "Unity Bank", "Heritage Bank", "Crest Finance Bank"]
APPS = ["QuickPay", "PayNest", "CashEasy", "LoanLeaf", "RupeeRush"]
FIRMS = ["Zentra Tech", "BrightPath Solutions", "NovaHire", "Orbit Consulting", "PeakWorks"]
WORDS = ["rahul", "priya", "care", "support", "help", "refund", "kyc", "verify",
         "pay", "secure", "bonus", "reward", "loan", "cash"]

# Five ways each scam type is commonly described. A gang tends to stick to 2 of them.
TEMPLATES = {
    "kyc": [
        "Received a call from someone claiming to be from {bank} saying my KYC had expired and my account would be blocked today. They sent a link and asked me to enter my card details and the OTP. {loss}",
        "An SMS said my {bank} account would be suspended in 24 hours unless I completed KYC. I clicked the link, filled in my details and shared the OTP. {loss}",
        "Caller pretended to be a {bank} officer, said my PAN was not linked and asked me to install a screen sharing app to fix it. {loss}",
        "I got a WhatsApp message with a KYC update form from {bank}. After I submitted my account number and OTP, money was debited. {loss}",
        "Someone said they were calling from {bank} head office about pending verification and kept me on the line until I read out the OTP. {loss}",
    ],
    "refund": [
        "I listed a product for sale online and the buyer said he would pay by UPI. He sent a collect request and told me to approve it to receive the money. {loss}",
        "Ordered food, it was never delivered, and I searched for customer care online. The person asked me to scan a QR code to get my refund. {loss}",
        "A caller said my payment had failed and a refund was pending. He told me to enter my UPI PIN to receive it. {loss}",
        "The seller asked for a small token amount by UPI, then said a mistake happened and asked me to send the amount again for a refund. {loss}",
        "I received a message about a cashback of Rs {small} and was asked to approve a request on my payment app. Instead money was deducted. {loss}",
    ],
    "job": [
        "I applied for a work from home job and a recruiter from {firm} asked for a registration fee of Rs {small}. After I paid, they stopped replying. {loss}",
        "Got a Telegram message offering easy money for liking videos. They asked me to deposit money to unlock higher tasks and then blocked me. {loss}",
        "An offer letter from {firm} arrived by email and I was asked to pay for training material and a laptop deposit. The company turned out to be fake. {loss}",
        "A person on WhatsApp promised daily payment for part time online work, collected a security deposit, and vanished. {loss}",
        "The HR of {firm} interviewed me over chat and asked for a processing fee before joining. After payment the number went unreachable. {loss}",
    ],
    "loan": [
        "I installed the {app} loan app and took a small loan. They accessed my contacts and are threatening to send morphed photos to my family unless I pay more. {loss}",
        "The {app} app asked for gallery and contact permissions, and after I repaid the loan they kept demanding extra money with abusive calls. {loss}",
        "Agents of {app} sent messages to my relatives calling me a fraud and demanded payment far above what I borrowed. {loss}",
        "I was offered an instant loan through {app} but first had to pay a processing charge. The loan never arrived. {loss}",
        "A loan app called {app} charged huge interest and started harassing me and my friends after one day of delay. {loss}",
    ],
}
SCAM_TYPES = list(TEMPLATES)

LOSS_PHRASES = ["I lost Rs {amount}.", "Rs {amount} was taken from my account.",
                "Total loss is around Rs {amount}.", "They took Rs {amount} in total.",
                "I am out of Rs {amount} now."]

# Small "habits" a gang repeats, which helps narrative matching.
SIGNATURES = [
    "The caller spoke in a hurried tone and kept calling me sir.",
    "He kept saying this was the last warning.",
    "The voice sounded like a recorded message at first.",
    "They insisted that I should not tell anyone in my family.",
    "The person introduced himself as Mr. Sharma.",
    "He pressured me to finish within ten minutes.",
    "They kept switching between Hindi and English.",
    "A woman first spoke and then passed the call to a senior officer.",
    "The profile picture showed a man in a suit.",
    "They sent a screenshot of a fake ID card.",
    "He said my case number was already registered.",
    "The messages had many spelling mistakes.",
]


# ----------------------------------------------------------------------------
# Small helper functions
# ----------------------------------------------------------------------------
def fresh_phone(rng):
    return "00" + str(rng.randint(10000000, 99999999))          # invalid in real life


def fresh_upi(rng):
    return f"{rng.choice(WORDS)}{rng.randint(10, 9999)}@fakebank"


def fresh_url(rng):
    return f"http://{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randint(1, 99)}.example"


def add_noise(text, rng, rate=0.03):
    """Randomly damages a few words (typos), like a rushed complaint."""
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 4 and rng.random() < rate:
            j = rng.randint(1, len(w) - 2)
            kind = rng.choice(["swap", "drop", "double"])
            if kind == "swap":
                w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            elif kind == "drop":
                w = w[:j] + w[j + 1:]
            else:
                w = w[:j] + w[j] + w[j:]
            words[i] = w
    return " ".join(words)


def write_narrative(rng, scam_type, template, bank, app, firm, signature=None):
    amount = rng.randint(2, 250) * 1000 + rng.choice([0, 0, 500])
    loss = rng.choice(LOSS_PHRASES).format(amount=f"{amount:,}")
    text = template.format(bank=bank, app=app, firm=firm,
                           small=rng.choice([500, 1000, 1500, 2500]), loss=loss)
    if signature:
        text += " " + signature
    return add_noise(text, rng)


def random_date(rng, start, end):
    return start + timedelta(days=rng.randint(0, (end - start).days))


# ----------------------------------------------------------------------------
# Main generator
# ----------------------------------------------------------------------------
def generate(seed, n_gangs, n_singles):
    rng = random.Random(seed)
    year_start, year_end = date(2026, 1, 1), date(2026, 7, 31)

    # --- Create the gangs (each one has its own habits and identifiers) ---
    gangs = []
    for g in range(n_gangs):
        scam = rng.choice(SCAM_TYPES)
        slow = rng.random() < 0.15                    # 15% of gangs operate for months
        duration = rng.randint(150, 200) if slow else rng.randint(20, 70)
        start = random_date(rng, year_start, date(2026, 6, 15))
        gangs.append({
            "id": f"G{g + 1:02d}",
            "scam": scam,
            "templates": rng.sample(TEMPLATES[scam], 2),
            "bank": rng.choice(BANKS), "app": rng.choice(APPS), "firm": rng.choice(FIRMS),
            "signature": rng.choice(SIGNATURES),
            "cities": rng.sample(CITIES, rng.randint(2, 4)),
            "phones": [fresh_phone(rng) for _ in range(rng.randint(1, 2))],
            "upis": [fresh_upi(rng) for _ in range(rng.randint(1, 2))],
            "urls": [fresh_url(rng)] if rng.random() < 0.6 else [],
            "start": start, "end": min(start + timedelta(days=duration), date(2026, 8, 31)),
            "size": rng.randint(3, 9),
        })

    hub_phones = [fresh_phone(rng) for _ in range(3)]   # popular numbers seen in many unrelated cases
    rows = []                                           # (date, city, scam, text, phone, upi, url, gang_id)

    # --- Reports that belong to gangs ---
    for gg in gangs:
        for _ in range(gg["size"]):
            phone = upi = url = ""
            if rng.random() > 0.15:                     # 15% of reports carry no identifier at all
                if rng.random() < 0.6:
                    phone = rng.choice(gg["phones"]) if rng.random() > 0.10 else fresh_phone(rng)  # burner phone
                if rng.random() < 0.5:
                    upi = rng.choice(gg["upis"])
                if gg["urls"] and rng.random() < 0.4:
                    url = rng.choice(gg["urls"])
            sig = gg["signature"] if rng.random() < 0.7 else None
            text = write_narrative(rng, gg["scam"], rng.choice(gg["templates"]),
                                   gg["bank"], gg["app"], gg["firm"], sig)
            rows.append((random_date(rng, gg["start"], gg["end"]), rng.choice(gg["cities"]),
                         gg["scam"], text, phone, upi, url, gg["id"]))

    # --- Unrelated reports (look-alikes and traps) ---
    all_gang_phones = [p for gg in gangs for p in gg["phones"]]
    all_gang_upis = [u for gg in gangs for u in gg["upis"]]
    for _ in range(n_singles):
        scam = rng.choice(SCAM_TYPES)
        phone = upi = url = ""
        if rng.random() < 0.55:
            r = rng.random()
            if r < 0.12:
                phone = rng.choice(hub_phones)           # trap: popular number
            elif r < 0.17:
                phone = rng.choice(all_gang_phones)      # trap: recycled number
            else:
                phone = fresh_phone(rng)
        if rng.random() < 0.45:
            upi = rng.choice(all_gang_upis) if rng.random() < 0.06 else fresh_upi(rng)
        if rng.random() < 0.30:
            url = fresh_url(rng)
        sig = rng.choice(SIGNATURES) if rng.random() < 0.10 else None   # decoy habit
        text = write_narrative(rng, scam, rng.choice(TEMPLATES[scam]), rng.choice(BANKS),
                               rng.choice(APPS), rng.choice(FIRMS), sig)
        rows.append((random_date(rng, year_start, year_end), rng.choice(CITIES),
                     scam, text, phone, upi, url, "NONE"))

    # Sort by date and give IDs, so IDs do not reveal which reports are linked
    rows.sort(key=lambda r: (r[0], r[3]))
    return rows


def save(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "reports.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["report_id", "date", "city", "category", "narrative", "phone", "upi_id", "url"])
        for i, r in enumerate(rows, 1):
            w.writerow([f"R{i:04d}", r[0].isoformat(), r[1], r[2], r[3], r[4], r[5], r[6]])
    with open(os.path.join(out_dir, "ground_truth.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["report_id", "gang_id"])
        for i, r in enumerate(rows, 1):
            w.writerow([f"R{i:04d}", r[7]])


def print_summary(rows):
    n = len(rows)
    true_pairs = easy = false_trap = 0
    for a, b in combinations(rows, 2):
        shared = any(a[k] and a[k] == b[k] for k in (4, 5, 6))
        linked = a[7] != "NONE" and a[7] == b[7]
        if linked:
            true_pairs += 1
            easy += shared
        elif shared:
            false_trap += 1
    print(f"Reports: {n}  (in gangs: {sum(r[7] != 'NONE' for r in rows)}, unrelated: {sum(r[7] == 'NONE' for r in rows)})")
    print(f"Truly linked pairs: {true_pairs}")
    print(f"  of which share an identifier (easy): {easy}")
    print(f"  with NO shared identifier (hard):    {true_pairs - easy}")
    print(f"Unrelated pairs that share an identifier (traps): {false_trap}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gangs", type=int, default=40)
    ap.add_argument("--singles", type=int, default=160)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    data = generate(args.seed, args.gangs, args.singles)
    save(data, args.out)
    print_summary(data)
    print(f"Saved to {args.out}/reports.csv and {args.out}/ground_truth.csv")
