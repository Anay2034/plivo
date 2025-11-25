"""
Data Generator for PII NER Training
Generates noisy STT-style transcripts with labeled PII entities.
"""

import json
import random
import argparse
from typing import List, Dict, Tuple

# ============ DATA POOLS ============

FIRST_NAMES = [
    "rahul", "priya", "amit", "neha", "vijay", "anita", "rajesh", "sunita",
    "deepak", "pooja", "sanjay", "kavita", "arun", "meera", "suresh", "divya",
    "ramesh", "anjali", "rohit", "sneha", "manish", "rekha", "nikhil", "swati",
    "arjun", "pallavi", "varun", "ishita", "karan", "nisha", "tushar", "ritu",
    "dhruv", "ananya", "rohan", "shruti", "aditya", "preeti", "akash", "shweta",
    "gaurav", "komal", "vishal", "archana", "mohit", "megha", "harsh", "jyoti",
    "dev", "simran", "aman", "tanya", "kunal", "ritika", "sahil", "kriti",
    "vivek", "manisha", "ajay", "sakshi"
]

LAST_NAMES = [
    "sharma", "verma", "gupta", "singh", "kumar", "patel", "reddy", "rao",
    "iyer", "nair", "menon", "pillai", "joshi", "desai", "mehta", "shah",
    "chopra", "kapoor", "malhotra", "khanna", "bhatia", "agarwal", "saxena",
    "mishra", "pandey", "dubey", "tiwari", "yadav", "chauhan", "thakur",
    "banerjee", "mukherjee", "chatterjee", "ghosh", "das", "sen", "bose",
    "roy", "dutta", "paul", "sinha", "prasad", "krishnan", "subramanian",
    "chaudhary", "rathore", "rajput", "solanki", "parikh", "trivedi"
]

CITIES = [
    "mumbai", "delhi", "bangalore", "hyderabad", "chennai", "kolkata", "pune",
    "ahmedabad", "jaipur", "lucknow", "kanpur", "nagpur", "indore", "thane",
    "bhopal", "visakhapatnam", "patna", "vadodara", "ghaziabad", "ludhiana",
    "agra", "nashik", "faridabad", "meerut", "rajkot", "varanasi", "srinagar",
    "aurangabad", "dhanbad", "amritsar", "allahabad", "ranchi", "coimbatore",
    "jabalpur", "gwalior", "vijayawada", "madurai", "guwahati", "chandigarh",
    "hubli", "mysore", "trichy", "bareilly", "aligarh", "moradabad", "gurgaon",
    "noida", "kochi", "trivandrum", "mangalore"
]

LOCATIONS = [
    "koramangala", "indiranagar", "whitefield", "electronic city", "hsr layout",
    "marathahalli", "jayanagar", "btm layout", "jp nagar", "banashankari",
    "bandra", "andheri", "powai", "worli", "juhu", "malad", "goregaon",
    "thane west", "navi mumbai", "vashi", "kharghar", "panvel",
    "connaught place", "karol bagh", "dwarka", "rohini", "saket", "vasant kunj",
    "greater kailash", "lajpat nagar", "hauz khas", "defence colony",
    "banjara hills", "jubilee hills", "gachibowli", "madhapur", "hitech city",
    "salt lake", "park street", "new town", "rajarhat", "howrah",
    "mg road", "brigade road", "church street", "residency road",
    "anna nagar", "t nagar", "adyar", "velachery", "omr", "ecr"
]

EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "rediffmail.com",
    "gmail.co", "yahoo.co.in", "hotmail.co.in", "rediffmail.co.in",
    "protonmail.com", "protonmail.in", "icloud.com", "live.com",
    "gmail.in", "yahoo.in", "outlook.in"
]

# ============ HELPER FUNCTIONS ============

def num_to_words(n: int) -> str:
    """Convert a single digit to word."""
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return words[n]

def format_phone_spoken(digits: str) -> str:
    """Format phone number in various STT styles."""
    style = random.choice(["digits", "words", "mixed", "grouped", "double"])
    
    if style == "digits":
        # Plain digits with optional spaces
        if random.random() < 0.5:
            return digits
        else:
            # Add space in middle
            mid = len(digits) // 2
            return f"{digits[:mid]} {digits[mid:]}"
    
    elif style == "words":
        # All spoken words
        return " ".join(num_to_words(int(d)) for d in digits)
    
    elif style == "mixed":
        # Mix of digits and words
        result = []
        for d in digits:
            if random.random() < 0.5:
                result.append(d)
            else:
                result.append(num_to_words(int(d)))
        return " ".join(result)
    
    elif style == "grouped":
        # Grouped like "98765 43210"
        if len(digits) == 10:
            return f"{digits[:5]} {digits[5:]}"
        return digits
    
    else:  # double
        # Use "double" for repeated digits
        result = []
        i = 0
        while i < len(digits):
            if i + 1 < len(digits) and digits[i] == digits[i+1]:
                result.append(f"double {num_to_words(int(digits[i]))}")
                i += 2
            else:
                if random.random() < 0.5:
                    result.append(num_to_words(int(digits[i])))
                else:
                    result.append(digits[i])
                i += 1
        return " ".join(result)

def format_credit_card_spoken(digits: str) -> str:
    """Format credit card in various STT styles."""
    style = random.choice(["words", "mixed", "grouped", "digits"])
    
    if style == "words":
        return " ".join(num_to_words(int(d)) for d in digits)
    elif style == "mixed":
        result = []
        for d in digits:
            if random.random() < 0.4:
                result.append(d)
            else:
                result.append(num_to_words(int(d)))
        return " ".join(result)
    elif style == "grouped":
        # Group as 4-4-4-4
        groups = [digits[i:i+4] for i in range(0, 16, 4)]
        return " ".join(groups)
    else:
        return digits

def format_email_spoken(email: str) -> str:
    """Format email in STT style."""
    style = random.choice(["spoken", "mixed", "normal"])
    
    if style == "normal":
        return email
    
    # Replace @ and . with spoken forms
    result = email
    if random.random() < 0.7:
        result = result.replace("@", " at ")
    if random.random() < 0.6:
        result = result.replace(".", " dot ")
    
    return result.strip()

def format_date_spoken(day: int, month: int, year: int) -> str:
    """Format date in various STT styles."""
    months = ["january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december"]
    
    style = random.choice(["dmy_slash", "dmy_dash", "spoken", "spoken_th", "d_of_month"])
    
    if style == "dmy_slash":
        return f"{day:02d}/{month:02d}/{year}"
    elif style == "dmy_dash":
        return f"{day:02d}-{month:02d}-{year}"
    elif style == "spoken":
        return f"{day} {months[month-1]} {year}"
    elif style == "spoken_th":
        suffix = "th"
        if day == 1 or day == 21 or day == 31:
            suffix = "st"
        elif day == 2 or day == 22:
            suffix = "nd"
        elif day == 3 or day == 23:
            suffix = "rd"
        return f"{day}{suffix} {months[month-1]} {year}"
    else:
        return f"{day} of {months[month-1]} {year}"

def generate_phone() -> str:
    """Generate Indian phone number."""
    # Indian mobile numbers start with 6-9
    first = random.choice(["6", "7", "8", "9"])
    rest = "".join(random.choices("0123456789", k=9))
    return first + rest

def generate_credit_card() -> str:
    """Generate 16-digit credit card number."""
    return "".join(random.choices("0123456789", k=16))

def generate_email(first_name: str, last_name: str) -> str:
    """Generate email from name."""
    domain = random.choice(EMAIL_DOMAINS)
    style = random.choice(["dot", "underscore", "direct", "initial"])
    
    if style == "dot":
        return f"{first_name}.{last_name}@{domain}"
    elif style == "underscore":
        return f"{first_name}_{last_name}@{domain}"
    elif style == "direct":
        return f"{first_name}{last_name}@{domain}"
    else:
        return f"{first_name[0]}.{last_name}@{domain}"

def generate_date() -> Tuple[int, int, int]:
    """Generate random date."""
    year = random.randint(2023, 2027)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return day, month, year

# ============ TEMPLATE GENERATORS ============

def gen_full_info() -> Dict:
    """Generate: name + city + phone + email + date"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    city = random.choice(CITIES)
    phone_raw = generate_phone()
    phone = format_phone_spoken(phone_raw)
    email_raw = generate_email(first, last)
    email = format_email_spoken(email_raw)
    day, month, year = generate_date()
    date = format_date_spoken(day, month, year)
    
    templates = [
        f"this is {name} from {city} my phone is {phone} and email is {email} we can meet on {date}",
        f"my name is {name} i am from {city} call me on {phone} or email {email} lets meet {date}",
        f"hi i am {name} living in {city} my number is {phone} email {email} available on {date}",
        f"hello this is {name} from {city} you can reach me at {phone} or {email} meeting on {date}",
    ]
    
    text = random.choice(templates)
    entities = []
    
    # Find spans
    name_start = text.find(name)
    if name_start >= 0:
        entities.append({"start": name_start, "end": name_start + len(name), "label": "PERSON_NAME"})
    
    city_start = text.find(city)
    if city_start >= 0:
        entities.append({"start": city_start, "end": city_start + len(city), "label": "CITY"})
    
    phone_start = text.find(phone)
    if phone_start >= 0:
        entities.append({"start": phone_start, "end": phone_start + len(phone), "label": "PHONE"})
    
    email_start = text.find(email)
    if email_start >= 0:
        entities.append({"start": email_start, "end": email_start + len(email), "label": "EMAIL"})
    
    date_start = text.find(date)
    if date_start >= 0:
        entities.append({"start": date_start, "end": date_start + len(date), "label": "DATE"})
    
    return {"text": text, "entities": entities}

def gen_name_phone() -> Dict:
    """Generate: name + phone"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    phone_raw = generate_phone()
    phone = format_phone_spoken(phone_raw)
    
    templates = [
        f"this is {name} my phone number is {phone} please call me tomorrow",
        f"my name is {name} and my contact number is {phone}",
        f"i am {name} you can call me on {phone}",
        f"hello {name} here my mobile is {phone} call anytime",
        f"this is {name} reach me at {phone} thanks",
    ]
    
    text = random.choice(templates)
    entities = []
    
    name_start = text.find(name)
    if name_start >= 0:
        entities.append({"start": name_start, "end": name_start + len(name), "label": "PERSON_NAME"})
    
    phone_start = text.find(phone)
    if phone_start >= 0:
        entities.append({"start": phone_start, "end": phone_start + len(phone), "label": "PHONE"})
    
    return {"text": text, "entities": entities}

def gen_name_email() -> Dict:
    """Generate: name + email"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    email_raw = generate_email(first, last)
    email = format_email_spoken(email_raw)
    
    templates = [
        f"email id of {name} is {email}",
        f"my name is {name} and my email is {email}",
        f"i am {name} please send mail to {email}",
        f"this is {name} email me at {email}",
        f"contact {name} at {email} for details",
    ]
    
    text = random.choice(templates)
    entities = []
    
    name_start = text.find(name)
    if name_start >= 0:
        entities.append({"start": name_start, "end": name_start + len(name), "label": "PERSON_NAME"})
    
    email_start = text.find(email)
    if email_start >= 0:
        entities.append({"start": email_start, "end": email_start + len(email), "label": "EMAIL"})
    
    return {"text": text, "entities": entities}

def gen_name_city_date() -> Dict:
    """Generate: name + city + date (travel)"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    city = random.choice(CITIES)
    day, month, year = generate_date()
    date = format_date_spoken(day, month, year)
    
    templates = [
        f"i am {name} travelling to {city} on {date}",
        f"this is {name} i will be in {city} on {date}",
        f"my name is {name} booking flight to {city} for {date}",
        f"hello {name} here need hotel in {city} on {date}",
        f"{name} planning visit to {city} around {date}",
    ]
    
    text = random.choice(templates)
    entities = []
    
    name_start = text.find(name)
    if name_start >= 0:
        entities.append({"start": name_start, "end": name_start + len(name), "label": "PERSON_NAME"})
    
    city_start = text.find(city)
    if city_start >= 0:
        entities.append({"start": city_start, "end": city_start + len(city), "label": "CITY"})
    
    date_start = text.find(date)
    if date_start >= 0:
        entities.append({"start": date_start, "end": date_start + len(date), "label": "DATE"})
    
    return {"text": text, "entities": entities}

def gen_credit_card() -> Dict:
    """Generate: name + credit card + optional date/email"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    cc_raw = generate_credit_card()
    cc = format_credit_card_spoken(cc_raw)
    city = random.choice(CITIES)
    
    day, month, year = generate_date()
    date = format_date_spoken(day, month, year)
    
    email_raw = generate_email(first, last)
    email = format_email_spoken(email_raw)
    
    templates = [
        f"my name is {name} i am from {city} my credit card number is {cc} and it expires on {date} you can email me on {email}",
        f"this is {name} my card number is {cc} expiry {date}",
        f"i am {name} please charge card {cc} thanks",
        f"hello {name} here credit card {cc} for payment",
        f"card details for {name} number {cc} valid till {date}",
    ]
    
    text = random.choice(templates)
    entities = []
    
    name_start = text.find(name)
    if name_start >= 0:
        entities.append({"start": name_start, "end": name_start + len(name), "label": "PERSON_NAME"})
    
    if city in text:
        city_start = text.find(city)
        if city_start >= 0:
            entities.append({"start": city_start, "end": city_start + len(city), "label": "CITY"})
    
    cc_start = text.find(cc)
    if cc_start >= 0:
        entities.append({"start": cc_start, "end": cc_start + len(cc), "label": "CREDIT_CARD"})
    
    if date in text:
        date_start = text.find(date)
        if date_start >= 0:
            entities.append({"start": date_start, "end": date_start + len(date), "label": "DATE"})
    
    if email in text:
        email_start = text.find(email)
        if email_start >= 0:
            entities.append({"start": email_start, "end": email_start + len(email), "label": "EMAIL"})
    
    return {"text": text, "entities": entities}

def gen_location_city() -> Dict:
    """Generate: location + city"""
    location = random.choice(LOCATIONS)
    city = random.choice(CITIES)
    
    templates = [
        f"the office is near {location} in {city} today",
        f"i live in {location} area of {city}",
        f"our branch is at {location} in {city}",
        f"meeting point is {location} in {city} tomorrow",
        f"address is {location} {city} please note",
    ]
    
    text = random.choice(templates)
    entities = []
    
    loc_start = text.find(location)
    if loc_start >= 0:
        entities.append({"start": loc_start, "end": loc_start + len(location), "label": "LOCATION"})
    
    city_start = text.find(city)
    if city_start >= 0:
        entities.append({"start": city_start, "end": city_start + len(city), "label": "CITY"})
    
    return {"text": text, "entities": entities}

def gen_name_location_city() -> Dict:
    """Generate: name + location + city"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    location = random.choice(LOCATIONS)
    city = random.choice(CITIES)
    
    templates = [
        f"my name is {name} i work in {location} in {city}",
        f"this is {name} from {location} area {city}",
        f"i am {name} my office is in {location} {city}",
        f"hello {name} here based in {location} near {city}",
    ]
    
    text = random.choice(templates)
    entities = []
    
    name_start = text.find(name)
    if name_start >= 0:
        entities.append({"start": name_start, "end": name_start + len(name), "label": "PERSON_NAME"})
    
    loc_start = text.find(location)
    if loc_start >= 0:
        entities.append({"start": loc_start, "end": loc_start + len(location), "label": "LOCATION"})
    
    city_start = text.find(city)
    if city_start >= 0:
        entities.append({"start": city_start, "end": city_start + len(city), "label": "CITY"})
    
    return {"text": text, "entities": entities}

def gen_negative() -> Dict:
    """Generate: no entities (negative examples)"""
    phrases = [
        "tomorrow problem delivery please information payment issue update balance complaint resolve checking plan status yesterday",
        "complaint order tomorrow ticket feedback please support balance issue today",
        "soon status fast help ticket plan problem service support please checking payment balance complaint tomorrow",
        "yes i need help with my order it is delayed",
        "please check my account balance and update status",
        "i want to cancel my subscription immediately",
        "when will my delivery arrive please update",
        "the service is not working properly please help",
        "i have a complaint about recent transaction",
        "need to speak with customer support urgently",
        "my package was damaged during shipping",
        "please refund my money as soon as possible",
        "the product quality is not as expected",
        "i want to change my plan to premium",
        "how do i reset my password please help",
        "the app is crashing every time i open it",
        "i need invoice for my recent purchase",
        "when is the sale starting this month",
        "please transfer me to billing department",
        "i have been waiting for thirty minutes already",
    ]
    return {"text": random.choice(phrases), "entities": []}

def gen_hinglish_style() -> Dict:
    """Generate: Hinglish/code-mixed style (stress test)"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    city = random.choice(CITIES)
    day, month, year = generate_date()
    date = format_date_spoken(day, month, year)
    
    templates = [
        f"haan so my naam is {name} and main rehte in {city} we will meet on {date}",
        f"arey mera name {name} hai aur main {city} mein hoon milte hain {date}",
        f"dekhiye {name} bol raha hoon {city} se call kar raha hoon {date} ko milenge",
        f"hello ji {name} this side from {city} lets connect on {date}",
    ]
    
    text = random.choice(templates)
    entities = []
    
    name_start = text.find(name)
    if name_start >= 0:
        entities.append({"start": name_start, "end": name_start + len(name), "label": "PERSON_NAME"})
    
    city_start = text.find(city)
    if city_start >= 0:
        entities.append({"start": city_start, "end": city_start + len(city), "label": "CITY"})
    
    date_start = text.find(date)
    if date_start >= 0:
        entities.append({"start": date_start, "end": date_start + len(date), "label": "DATE"})
    
    return {"text": text, "entities": entities}

def gen_complex_cc_phone_email() -> Dict:
    """Generate: complex with CC + phone + email (stress test)"""
    cc_raw = generate_credit_card()
    cc = format_credit_card_spoken(cc_raw)
    phone_raw = generate_phone()
    phone = format_phone_spoken(phone_raw)
    
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    email_raw = generate_email(first, last)
    email = format_email_spoken(email_raw)
    
    templates = [
        f"uh actually my old card number maybe is {cc} i am not sure and my new phone is {phone} also send email to {email} please",
        f"card number is {cc} phone {phone} email {email} please process",
        f"details are card {cc} mobile {phone} mail {email} thanks",
    ]
    
    text = random.choice(templates)
    entities = []
    
    cc_start = text.find(cc)
    if cc_start >= 0:
        entities.append({"start": cc_start, "end": cc_start + len(cc), "label": "CREDIT_CARD"})
    
    phone_start = text.find(phone)
    if phone_start >= 0:
        entities.append({"start": phone_start, "end": phone_start + len(phone), "label": "PHONE"})
    
    email_start = text.find(email)
    if email_start >= 0:
        entities.append({"start": email_start, "end": email_start + len(email), "label": "EMAIL"})
    
    return {"text": text, "entities": entities}

def gen_order_id_negative() -> Dict:
    """Generate: contains numbers but not PII (negative)"""
    order_id = random.randint(100000, 999999)
    templates = [
        f"this is regarding order id {order_id} and i checked it two three times already",
        f"my order number is {order_id} please check status",
        f"reference number {order_id} for complaint",
        f"ticket id {order_id} still not resolved",
        f"booking reference {order_id} please confirm",
    ]
    return {"text": random.choice(templates), "entities": []}

# ============ MAIN GENERATOR ============

def generate_dataset(n_samples: int, include_stress: bool = False) -> List[Dict]:
    """Generate a dataset with n_samples."""
    generators = [
        (gen_full_info, 0.20),
        (gen_name_phone, 0.15),
        (gen_name_email, 0.10),
        (gen_name_city_date, 0.10),
        (gen_credit_card, 0.10),
        (gen_location_city, 0.08),
        (gen_name_location_city, 0.07),
        (gen_negative, 0.12),
        (gen_order_id_negative, 0.08),
    ]
    
    if include_stress:
        generators.extend([
            (gen_hinglish_style, 0.05),
            (gen_complex_cc_phone_email, 0.05),
        ])
    
    # Normalize weights
    total_weight = sum(w for _, w in generators)
    generators = [(g, w/total_weight) for g, w in generators]
    
    data = []
    for i in range(n_samples):
        # Weighted random selection
        r = random.random()
        cumulative = 0
        selected_gen = generators[0][0]
        for gen, weight in generators:
            cumulative += weight
            if r <= cumulative:
                selected_gen = gen
                break
        
        sample = selected_gen()
        # Reorder keys to match original format: id, text, entities
        ordered_sample = {
            "id": f"utt_{i:04d}",
            "text": sample["text"],
            "entities": sample["entities"]
        }
        data.append(ordered_sample)
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate PII NER training data")
    parser.add_argument("--train_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--dev_samples", type=int, default=200, help="Number of dev samples")
    parser.add_argument("--train_output", default="data/train_generated.jsonl")
    parser.add_argument("--dev_output", default="data/dev_generated.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Generate training data
    print(f"Generating {args.train_samples} training samples...")
    train_data = generate_dataset(args.train_samples, include_stress=True)
    
    with open(args.train_output, "w", encoding="utf-8") as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved to {args.train_output}")
    
    # Generate dev data with different seed
    random.seed(args.seed + 1000)
    print(f"Generating {args.dev_samples} dev samples...")
    dev_data = generate_dataset(args.dev_samples, include_stress=True)
    
    # Renumber IDs for dev
    for i, sample in enumerate(dev_data):
        dev_data[i] = {
            "id": f"dev_{i:04d}",
            "text": sample["text"],
            "entities": sample["entities"]
        }
    
    with open(args.dev_output, "w", encoding="utf-8") as f:
        for sample in dev_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved to {args.dev_output}")
    
    # Print stats
    print("\n--- Statistics ---")
    for name, data in [("Train", train_data), ("Dev", dev_data)]:
        entity_counts = {}
        for sample in data:
            for ent in sample["entities"]:
                entity_counts[ent["label"]] = entity_counts.get(ent["label"], 0) + 1
        print(f"{name}: {len(data)} samples")
        for label, count in sorted(entity_counts.items()):
            print(f"  {label}: {count}")

if __name__ == "__main__":
    main()