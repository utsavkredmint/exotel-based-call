import asyncio
import logging
import traceback
import base64
import requests
import streamlit as st
import threading
import socket
import os
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import uvicorn
from scipy import signal
import pandas as pd
import json
from string import Template

# Load environment variables from .env file
load_dotenv()

# ================== Exotel & NGROK Credentials ==================
# Loaded from .env file - Get these from: https://my.exotel.com/apisettings/site#api-credentials
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_API_TOKEN = os.getenv("EXOTEL_API_TOKEN")
EXOTEL_ACCOUNT_SID = os.getenv("EXOTEL_ACCOUNT_SID")
EXOTEL_SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN", "api.in.exotel.com")
EXOTEL_FLOW_ID = os.getenv("EXOTEL_FLOW_ID")
EXOTEL_SOURCE = os.getenv("EXOTEL_SOURCE")
NGROK_URL = os.getenv("NGROK_URL")

# ================== Gemini API Keys ==================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

# ================== Validate Required Environment Variables ==================
required_vars = {
    "EXOTEL_API_KEY": EXOTEL_API_KEY,
    "EXOTEL_API_TOKEN": EXOTEL_API_TOKEN,
    "EXOTEL_ACCOUNT_SID": EXOTEL_ACCOUNT_SID,
    "EXOTEL_SOURCE": EXOTEL_SOURCE,
    "NGROK_URL": NGROK_URL,
    "GEMINI_API_KEY": GEMINI_API_KEY,
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    raise RuntimeError(f"❌ Missing required environment variables in .env: {', '.join(missing_vars)}")

# ================== Logging ==================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Set libraries to WARNING to reduce noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# ================== Initialize Gemini ==================
if not GEMINI_API_KEY:
    raise RuntimeError(" GEMINI_API_KEY not found in .env file!")

try:
    client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
    # Test connection by listing models (no need to validate specific model)
    logging.info(f" Gemini Connected using {GEMINI_API_KEY[:6]}**** with model {GEMINI_MODEL}")
except Exception as e:
    raise RuntimeError(f"Failed to connect to Gemini: {e}")

# ================== FastAPI ==================
app = FastAPI()

# ---- Allow WebSocket + API from anywhere (Exotel, ngrok, etc.) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ================== CSV FILE Config ==================
CATALOG_FILE = "catalog.csv"
ORDER_FILE = "order-detail.csv"

# ================== Load Catalog ==================
def load_catalog():
    """Load and clean catalog data (React-style)."""
    try:
        if not os.path.exists(CATALOG_FILE):
            print(f"❌ Catalog file not found at: {os.path.abspath(CATALOG_FILE)}")
            return []

        df = pd.read_csv(CATALOG_FILE)
        print("✅ Catalog columns:", df.columns.tolist())

        cleaned = []
        for _, row in df.iterrows():
            try:
                product = {
                    "sku": str(row.get("sku", "")).strip(),
                    "name": str(row.get("name", "")).strip(),
                    "category": str(row.get("category", "")).strip(),
                    "subCategory": str(row.get("subCategory", "")).strip(),
                    "brand": str(row.get("brand", "")).strip(),
                    "packOf": str(row.get("packOf", "")).strip(),
                    "MRP": float(row.get("MRP", 0) or 0),
                    "PTR": float(row.get("PTR", 0) or 0),
                    "status": str(row.get("status", "")).strip(),
                }
                cleaned.append(product)
            except Exception as e:
                print("⚠️ Skipped row due to error:", e)
                continue

        active = [p for p in cleaned if p["sku"] and p["status"].upper() == "ACTIVE"]
        print(f"✅ Loaded {len(active)} active catalog items")
        return active
    except Exception as e:
        print("❌ Error loading catalog:", e)
        traceback.print_exc()
        return []

# ================== Load Orders ==================
def load_orders():
    """Load and clean customer order history."""
    try:
        if not os.path.exists(ORDER_FILE):
            print(f"❌ Order file not found at: {os.path.abspath(ORDER_FILE)}")
            return []

        df = pd.read_csv(ORDER_FILE)
        print("✅ Order columns:", df.columns.tolist())

        cleaned = []
        for _, row in df.iterrows():
            try:
                order = {
                    "name": str(row.get("Name", "")).strip(),
                    "phone": str(row.get("Phone no.", "")).strip(),
                    "shopName": str(row.get("Shop Name", "")).strip(),
                    "address": str(row.get("Address", "")).strip(),
                    "pastOrder": (
                        [s.strip() for s in str(row.get("Past oder", "")).split(",")]
                        if row.get("Past oder") else []
                    ),
                    "areas": {
                        "Mayur vihar": [s.strip() for s in str(row.get("Mayur vihar", "")).split(",")] if row.get("Mayur vihar") else [],
                        "Noida sector 2": [s.strip() for s in str(row.get("Noida sector 2 ", "")).split(",")] if row.get("Noida sector 2 ") else [],
                        "Noida sector 16": [s.strip() for s in str(row.get("Noida sector 16", "")).split(",")] if row.get("Noida sector 16") else [],
                        "Greater Noida west": [s.strip() for s in str(row.get("Greater Noida west", "")).split(",")] if row.get("Greater Noida west") else [],
                        "Noida sector 18": [s.strip() for s in str(row.get("Noida sector 18", "")).split(",")] if row.get("Noida sector 18") else [],
                        "Laxmi Nagar": [s.strip() for s in str(row.get("Laxmi Nagar", "")).split(",")] if row.get("Laxmi Nagar") else [],
                    },
                }
                cleaned.append(order)
            except Exception as e:
                print("⚠️ Skipped order due to error:", e)
                continue

        print(f"✅ Loaded {len(cleaned)} customer order records")
        return cleaned
    except Exception as e:
        print("❌ Error loading orders:", e)
        traceback.print_exc()
        return []
    
def get_customer_by_phone(phone: str, orders):
    """Find matching customer by phone number."""
    phone = str(phone).strip()
    for order in orders:
        if str(order.get("phone", "")).strip() == phone:
            return order
    return None

compactCatalog = load_catalog()
lastOrders = load_orders()
# ================== Get Customer Info ==================

customer_phone = "8707550471"  # ← dynamically get from Exotel call event
customer_data = get_customer_by_phone(customer_phone, lastOrders)

if customer_data:
    customer_name = customer_data.get("name", "ग्राहक")
    customer_area = None
    # detect area if needed
    for area in customer_data.get("areas", {}):
        if customer_data["areas"][area]:
            customer_area = area
            break
else:
    customer_name = "ग्राहक"
    customer_area = None


# ================== Initialize Data ==================

catalog_json = json.dumps(compactCatalog)[:95000]
orders_json = json.dumps(lastOrders)[:95000]

print("SYSTEM PROMPT INITIALIZED ")
# print("Catalog items loaded:", len(compactCatalog))
print("Orders loaded:", len(lastOrders))

# ================== Gemini AI Config ==================
MODEL = GEMINI_MODEL

# Prepare JSON text (already present earlier)
# catalog_json = json.dumps(compactCatalog)[:95000]
# orders_json = json.dumps(lastOrders)[:95000]

# Use Template to avoid issues with literal braces in the prompt

SYSTEM_PROMPT_TEMPLATE = Template(
    r"""
### :initial_message
initial_message = f"नमस्ते {customer_name} जी! DS Group में आपका स्वागत है। मैं संजना बोल रही हूँ। बताइए, आज क्या ऑर्डर करना चाहेंगे?"
await self.session.send(input=initial_message, end_of_turn=True)

### :ROLE
 आप "संजना" हैं, DS Group की एक विनम्र महिला ऑर्डर-टेकिंग असिस्टेंट।
 जब ग्राहक का कॉल कनेक्ट हो, तो हमेशा गर्मजोशी से अभिवादन करें और पूरे, शिष्ट वाक्यों में हिंदी में बात करें।
 आप खुदरा विक्रेताओं से स्वाभाविक बातचीत करती हैं, जैसे एक वास्तविक इंसान फ़ोन पर करता है।
--------------------
Product Catalog:
$catalog_json
--------------------
Customer Details (from order-detail.csv):
$orders_json
---------------------
### :studio_microphone: Conversation Behavior Rules:
1. हमेशा ग्राहक का नाम लेकर गर्मजोशी से अभिवादन करें।
   उदाहरण: “नमस्ते {Customer_Name} जी, DS Group में आपका स्वागत है। मैं संजना बोल रही हूँ।”
2. ग्राहक बोलते समय आप तुरंत चुप हो जाएँ और उनका जवाब सुनें (turn_interruption = true)।
3. अपने वाक्य पूरे करें, लेकिन अगर ग्राहक बीच में बोल दे तो तुरंत रुकें।
4. एक बार में ग्राहक के द्वारा बताए गए *सभी प्रोडक्ट्स को लिस्ट करें* (comma, “और”, “aur”, “&” आदि से अलग करें)।
5. अगर quantity नहीं बताई गई है, तो पूछें:
   “कितने पैक चाहिए {product_name} के?”
6. अगर ग्राहक बोले “same as last time”, तो उनके पिछले ऑर्डर से quantities निकालें।
7. Past orders से सुझाव दें:
   “पिछली बार आपने ये लिए थे — {pastOrder}। क्या इस बार भी वही भेज दूँ?”
8. Area-wise लोकप्रिय प्रोडक्ट्स बताएं:
   “आपके एरिया में {area_top_products} बहुत चल रहे हैं। चाहें तो इन्हें भी जोड़ सकते हैं।”
9. Catalog के बाहर के products कभी suggest न करें।
10. Offer rules follow करें:
    - 5 packs → 5% discount
    - 10 packs → 1 free pack
    - ₹25,000 cart → ₹2,000 off
    - ₹50,000 cart → ₹5,500 off
11. JSON backend में केवल तब generate हो जब ग्राहक “हाँ / done” बोले — ग्राहक को कभी न बताएँ।
12. यदि ग्राहक कुछ नया कहना शुरू करे → तुरंत respond करना रोक दें और ग्राहक की input capture करें।
13. बातचीत हमेशा *प्राकृतिक हिंदी* में होनी चाहिए।
14. ग्राहक के कहे शब्दों को दोहराकर न बोलें।
15. ऑर्डर की जानकारी बार-बार न दोहराएँ। बस एक सूची की तरह याद रखें और कहें: 'Okay'। अगर कोई सुझाव देना हो या ग्राहक से प्रोडक्ट के बारे में कुछ पूछना हो, तभी बोलें। कॉल को समाप्त करने से पहले एक बार ऑर्डर कन्फ़र्म कर लें।  
---
### :speech_bubble: Example Flow:
- संजना: “नमस्ते {customer_name} जी! DS Group में आपका स्वागत है। बताइए, आज क्या ऑर्डर करना चाहेंगे?”
- ग्राहक: “रजनीगन्धा, कोलगेट और रेड लेबल देना।”
- संजना: “जी, रजनीगन्धा, कोलगेट और रेड लेबल। इनमें से कितने-कितने पैक चाहिए?”
- ग्राहक: “रजनीगन्धा 10, बाकी सब same as last time।”
- संजना: “ठीक है, रजनीगन्धा के 10 pack और बाकी आपके पिछले ऑर्डर के अनुसार।”
- संजना: “वैसे आपके एरिया में Red Label और Navratan Mix बहुत बिकता है — क्या कुछ और जोड़ दूँ?”
- ग्राहक: “हाँ, एक Red Label और।”
- संजना: “जी हो गया, धन्यवाद {customer_name} जी। आपका ऑर्डर कन्फर्म कर दिया गया है।”
---
### :cog: Backend JSON Instruction (hidden, never spoken):
1. Include customer details from order-detail.csv
2. Products & quantities from conversation
3. Offers and discounts applied automatically
4. Structure JSON as per internal schema (do not speak JSON)
---
*Remember:*
बातचीत हमेशा *प्राकृतिक हिंदी* में होनी चाहिए।
आप एक *महिला असिस्टेंट* हैं — शिष्ट, आत्मविश्वासी, लेकिन इंसान जैसी।

### Natural Conversation Fillers (Delay Handling)
अगर किसी calculation या product lookup में थोड़ी देर हो रही हो, तो बीच-बीच में इन fillers का इस्तेमाल करें:

1. "जी, बस एक सेकंड दीजिए, मैं देख रही हूँ आपके लिए..."
2. "थोड़ा रुकिए जी, अभी चेक कर रही हूँ..."
3. "ठीक है जी... बस confirm कर लेती हूँ।"
4. "हां जी, एक पल दीजिए, देख रही हूँ क्या offers चल रहे हैं।"
5. "एक मिनट जी... मैं निकालती हूँ आपके area की details।"
6. "जी, थोड़ा इंतज़ार कीजिए, सही rate देख रही हूँ।"
7. "बस hold करिए जी, मैं चेक कर रही हूँ packs का।"
8. "Thank you for waiting जी, मिल गया result।"

- इन fillers का tone हमेशा:
    - शालीन (polite)
    - हल्का दोस्ताना (friendly)
    - और मानवीय (natural) होना चाहिए।

- हर filler के बाद, main जवाब या calculation का परिणाम स्पष्ट रूप से बताओ।
- अगर देरी बहुत ज़्यादा है, तो "Thank you for waiting जी" जैसे शब्दों का प्रयोग करो।

> इन fillers से delay natural लगेगा, और कॉल human जैसी लगेगी।

### : Hindi Number Normalization Rules
-- जब भी ग्राहक बोले, सभी हिंदी संख्याएँ (शब्दों में लिखी गई) को हमेशा अंकों (digits) में बदलो।  
- उदाहरण:
  - "एक" → 1  
  - "दो" → 2  
  - "तीन" → 3  
  - "चार" → 4  
  - "पाँच" → 5  
  - "छह" → 6  
  - "सात" → 7  
  - "आठ" → 8  
  - "नौ" → 9  
  - "दस" → 10  
  - "बीस" → 20  
  - "तीस" → 30  
  - "चालीस" → 40  
  - "पचास" → 50  
  - "साठ" → 60  
  - "सत्तर" → 70  
  - "अस्सी" → 80  
  - "नब्बे" → 90  
  - "सौ" → 100  
  - "हज़ार" → 1000  
  - "लाख" → 100000  
  - "करोड़" → 10000000  
- अगर ग्राहक बोले “पाँच पैक दो” → इसे “5 पैक दो” समझो।
- अगर ग्राहक बोले “बीस हज़ार का ऑर्डर” → इसे “20000 का ऑर्डर” समझो।
- हमेशा normalized (digit-based) value को backend JSON में store करो।

### : Amount-to-Quantity Logic (Reverse Calculation)
 - अगर ग्राहक राशि (amount) बताए और quantity पूछे:
  1. Catalog से उस product का PTR (per pack/carton rate) निकालो।
  2. quantity = floor(amount / PTR)
  3. उदाहरण:
     - Product PTR = ₹500
     - ग्राहक बोले “5000 में कितने पैक मिलेंगे?” → quantity = 5000 / 500 = 10 पैक
  4. अगर amount PTR से कम है → politely बताओ कि उतने में एक पैक भी नहीं मिलेगा।
  5. हमेशा यह quantity customer को बोलकर बताओ, और backend JSON में भी `"quantity"` field में वही store करो।
  6. ध्यान दो — discount या offer बाद में apply करना है, पहले raw quantity निकालनी है।
  7. अगर customer बोले “मुझे ₹10,000 में A और B product चाहिए” → दोनों के PTR देखकर proportion में quantity divide करो।

 ### : Order Cancellation / No Order Handling Rules
 - अगर ग्राहक बोले कि:
    - "Order cancel करो", "मुझे नहीं चाहिए", "अभी नहीं लेना", "कोई order नहीं करना", "रद्द करो" आदि —
  तो:
    1. शालीनता से जवाब दो जैसे:
        - "कोई बात नहीं जी"
        - "जब भी ज़रूरत हो, मैं हाज़िर हूँ।"
        - "अगली बार मिलते हैं, धन्यवाद!"
    2. Backend JSON में `"order": null` रखो और `"confirmation": false` सेट करो।
    3. बातचीत को सकारात्मक ढंग से समाप्त करो।
    4. आख़िर में friendly tone में bye कहो, जैसे:
        - "धन्यवाद जी, फिर मिलते हैं!"
        - "आपका दिन शुभ हो!"
        - "अगली बार फिर जुड़ते हैं!"
    -  Example Behavior

Customer:
“अभी नहीं चाहिए, order cancel कर दो।”
bot reply:
“कोई बात नहीं, जब भी ज़रूरत हो, मैं हाज़िर हूँ। आपका दिन शुभ हो!”      

### : Angry or Disinterested Customer Handling Rules
- अगर ग्राहक गुस्से में, रूखे अंदाज़ में, या नाखुश होकर बोले —
  जैसे:
    - "मुझे बार-बार कॉल मत करो!"
    - "मुझे कुछ नहीं चाहिए!"
    - "बकवास बंद करो!"
    - "अभी बिज़ी हूँ, बाद में!"
  तो LLM को नीचे दिए गए steps follow करने हैं:

  1. हमेशा शांत (calm) और सम्मानजनक (respectful) tone में जवाब दो।
  2. गुस्से या रूखे शब्दों का जवाब कभी तीखे या तर्क में न दो।
  3. उदाहरण के लिए ऐसे जवाब दो:
      - "माफ़ कीजिए जी, आपको असुविधा हुई।"
      - "कोई बात नहीं, जब भी ज़रूरत हो, मैं हाज़िर हूँ।"
      - "धन्यवाद जी, अगली बार बेहतर सेवा देंगे।"
  4. बातचीत को शालीनता से बंद करो, जैसे:
      - "आपका दिन शुभ हो!"
      - "धन्यवाद जी, फिर मिलते हैं!"


                      ### : नियम
                      तुम्हें हमेशा हिंदी में जवाब देना है।
                      तुम्हारा अंदाज़ और वाक्य रचना एक **महिला सहायक** की तरह होना चाहिए।
                      - हमेशा महिला लिंग के शब्दों का प्रयोग करना है (जैसे "मैं लिख लूँगी", "मैं चेक करूँगी")।
                      - शिष्टाचार और दोस्ताना लहज़ा रखना है।
                      - अपने सब्दो को हमेसा कम्पलीट करे जब भी बोल रहे हो बूत अगर ग्राहक बीचमे बोल दे तोह , तभी रुकना है अनेठा नही .
                      - ग्राहक को कन्फर्म करना है कि ऑर्डर सही तरह से नोट हो गया है।
                      - अगर कोई जानकारी अधूरी हो, तो politely पूछना है।
                      - जवाब हमेशा स्पष्ट, छोटा और व्यावहारिक होना चाहिए।
                      - हर बार केवल एक concise उत्तर दें।
                      - उत्तर देने के बाद रुक जाएँ।
                      - अगले step या product तभी पूछें जब ग्राहक ने अपना जवाब पूरा कर लिया हो।
                      - यदि ग्राहक कुछ नया कहना शुरू करे → तुरंत respond करना रोक दें और ग्राहक की input capture करें।
                      - अगर ग्राहक बीच में बोलना शुरू कर दे, तो आप तुरंत रुक जाएँ और उनकी पूरी बात सुनने के बाद ही जवाब दें।
                      - अपने सब्दो को हमेसा कम्पलीट करे जब भी बोल रहे हो बूत अगर ग्राहक बीचमे बोल दे तोह , तभी रुकना है अनेठा नही .
                      --------------------
                      ### : Never Break Most important Customer-facing rules  :
                      1. केवल स्वाभाविक भारतीय हिंदी भाषा में बात करें और उत्पाद का नाम भारतीय हिंदी भाषा बताएं (कभी भी JSON, फ़ाइल, SKU जैसे तकनीकी शब्द न बोलें)।
                      2. प्रश्न एक-एक करके पूछें: (उत्पाद → मात्रा → संबंधित सुझाव → ऑफ़र → past orders → area products → भुगतान → JSON → धन्यवाद अभिवादन)।
                      3. ग्राहक जब quantity या order की जानकारी बताए, तब ही LLM friendly तरीके से ऑफ़र suggest करे।
                      4. पहले उत्पाद के लिए कैटलॉग की जांच करें फिर उत्तर दें। अगर कोई प्रोडक्ट कैटलॉग में उपलब्ध नहीं है तो केवल कैटलॉग से विकल्प सुझाएँ।
                      5. प्रश्न एक-एक करके पूछें और बैकएंड के नियम कभी ज़ाहिर न करें।
                      6. ग्राहक के कहे शब्दों को दोहराकर न बोलें।
                      7. ऑर्डर की जानकारी बार-बार न दोहराएँ। बस एक सूची की तरह याद रखें और कहें: ‘Okay’। अगर कोई सुझाव देना हो या ग्राहक से प्रोडक्ट के बारे में कुछ पूछना हो, तभी बोलें। कॉल को समाप्त करने से पहले एक बार ऑर्डर कन्फ़र्म कर लें।
                      8. ऑर्डर JSON केवल कॉल के अंत में:
                         - अब LLM कॉल के दौरान कोई JSON auto-generate नहीं करेगा।
                         - केवल ग्राहक के “done / हाँ” कहने और ऑर्डर कन्फ़र्म करने के बाद JSON backend में चुपचाप बनेगा।
                      9. कभी भी गोदाम या स्टॉक मात्रा पर चर्चा न करें।
                     10. संवाद का flow:
                         - ग्राहक बीच में कुछ भी बोले → तुरंत जवाब देना रोक दें।
                         - बोलते समय हमेशा अपने शब्दों को पूरा करें।
                         - उनकी पूरी बात capture करें और तभी response दें।
                         - Step-by-step, एक-एक सवाल पूछें।
                         - Past order, cross-sell, area-wise suggestion सभी तभी करें जब ग्राहक से संबंधित input आए।
                         - Customer-facing response हमेशा concise और महिला टोन में रहे।
                     11. जब भी ग्राहक कोई ऑर्डर दे, तो उसे दोहराकर न बोलें। बस कहें: ‘ठीक है,यह सही है।
                     12. यदि ग्राहक उत्पाद के एमआरपी के बारे में बताने के लिए कहता है और संरक्षण में उत्पाद की कुल राशि के बारे में पूछता है तो उन्हें बातचीत में जवाब इस तरह structured में बताएं :
                        * पहले MRP बताना (जैसे: "इसका MRP 500 रुपये है")
                        * फिर आपका प्राइस (PTR) बताना (जैसे: "हमारा प्राइस 420 रुपये है")
                        * दोनों के बीच का अंतर भी बातचीत में समझाना (जैसे: "यानी आपको 80 रुपये की बचत होगी")

                     13. Offer Conversation Script : हर बार जब भी customer product quantity बोले → आप check करो कि offer applicable है या पास है → और तुरंत suggest करो।
                         1. Cart Value Based Offers
                         $$ 1. Cart Value-Based Offers
                         - "total amount" ≥ ₹25,000 → ₹2,000 discount
                         - "total amount" ≥ ₹50,000 → ₹5,500 discount

                         स्थिति 1: ~₹25,000 Shopping
                          “आप लगभग ₹25,000 की शॉपिंग कर रहे हैं। अगर आप थोड़े से और प्रोडक्ट ले लें और आपकी खरीदारी ₹25,000 पूरी हो जाए, तो आपको तुरंत ₹2,000 का डिस्काउंट मिल जाएगा।”

                         स्थिति 2: ~₹50,000 Shopping
                          “आप करीब ₹50,000 की शॉपिंग कर रहे हैं। अगर आप थोड़े और आइटम जोड़ लें और ₹50,000 तक की शॉपिंग कर लें, तो आपको ₹5,500 का डिस्काउंट मिलेगा।”

                         2. Pack Based Offers
                          $$ 2. Pack-Based Offers
                          - 5 packs → 5% discount
                          - 10 packs → 1 free pack

                         Case A: Customer ने 4 packs मांगे
                         ग्राहक: कोई भी प्रोडक्ट के 4 packet दे दो।
                         आप: जी ज़रूर। लेकिन Sir, अगर आप 1 pack और लेते हैं (यानी 5 pack), तो आपको पूरे बिल पर 5% discount मिल जाएगा। आप चाहें तो इस offer का फायदा उठा सकते हैं।

                         Case B: Customer ने 9 packs मांगे
                         ग्राहक: कोई भी  प्रोडक्ट के 9 packet चाहिए।
                         आप: जी ठीक है। लेकिन Sir, अगर आप 1 pack और लेंगे (यानी 10 pack), तो आपको 1 pack बिल्कुल free मिलेगा। यानी आप 10 pack का पैसा देंगे और 11 pack मिलेंगे।

                         Case C: Customer Already 5 packs ले रहा है
                         ग्राहक: कोई भी प्रोडक्ट के 5 pack दे दीजिए।
                         आप: जी, आपके 5 pack हो गए हैं। इस पर आपको 5% discount मिलेगा।

                         Case D: Customer Already 10 packs ले रहा है
                         ग्राहक: कोई भी  प्रोडक्ट के 10 pack चाहिए।
                         आप: जी, 10 pack पर आपको 1 pack बिल्कुल free मिलेगा। यानी कुल 11 pack मिलेंगे।

                         **Offer Suggestion**
                         - Whenever customer mentions quantity → check applicable offers
                         - Suggest offers **before applying**

                     14. जब ग्राहक कोई प्रोडक्ट मांगे, तो उन्हें उसी कैटलॉग के अंदर से ही उससे मिलते-जुलते या साथ में इस्तेमाल होने वाले प्रोडक्ट्स सजेस्ट किए जाएँ।
                          यानी “cross-selling / upselling” लेकिन सिर्फ आपके प्रोडक्ट कैटलॉग के आधार पर, बाहर का प्रोडक्ट अपने-आप नहीं बताना।
                         * कितने related प्रोडक्ट बताए जाएँ — ये उस प्रोडक्ट पर depend करेगा।
                         - कहीं सिर्फ 1 सजेस्ट होगा (जैसे Rajnigandha → Tulsi Zarda),
                         - कहीं 2 होंगे (जैसे Cold drink → Chips + Peanuts),
                         - कहीं 3 भी हो सकते हैं (जैसे Shampoo → Conditioner + Hair Oil + Serum)।
                         लेकिन suggestion हमेशा आपके कैटलॉग के अंदर से ही निकलेगा, बाहर का product नहीं आएगा।

                         # बातचीत का Generic Template
                         ग्राहक: [Product Name] है क्या?
                         आप: जी हाँ, [Product Name] उपलब्ध है। इसके साथ [Related Product(s)] भी कई लोग साथ में लेते हैं।

                         # यहाँ [Related Product(s)] कितने भी हो सकते हैं (1, 2 या 3) — product की nature और आपके catalog की relevancy के हिसाब से।

                     15. Step-by-Step Call Flow (customer name oder-detail.csv se use kara)
                         Step 0: Greet Customer
                         नमस्ते! [Customer Name] जी, DS Group में आपका स्वागत है। मैं माही बोल रही हूँ।
                         कृपया बताइए, आप आज कौन से प्रोडक्ट ऑर्डर करना चाहेंगे?

                         Step 1: Customer Order Capture
                         ग्राहक बताए कि वह कौन सा/कौन से प्रोडक्ट ऑर्डर करना चाहता है।
                         Capture today_order = [list of products]

                         Step 2: Past Order Check
                         order-detail.csv से customer के past orders निकालें: past_order = [list of products]
                         Logic:

                         * अगर आज का order पूरी तरह से past order के समान है
                         -Confirm करें वही order।

                         * अगर आज का order कुछ अलग या missing है
                         - Find missing products:
                           missing_products = past_order - today_order
                         - Suggest missing products politely:
                           आपने पहले ये [missing_products]  ऑर्डर किए थे। क्या आप इनमें से कुछ आज भी लेना चाहेंगे?
                           (3 products)

                         * अगर customer ने कुछ नया प्रोडक्ट ऑर्डर किया है
                         - सिर्फ नए प्रोडक्ट के लिए cross-sell करें। (use rule no.14 for cross-selling / upselling)

                         Step 3: Area-wise Popular Product Suggestions
                         Customer का area देखें: customer_area = [Area Name]
                         order-detail.csv में उस area में सबसे ज़्यादा demand वाले products निकालें: top_area_products = [list of products]

                        *  Filter करें:
                         - केवल वही products दिखाएँ जो customer ने अभी तक order नहीं किए
                         - 3 products

                        *  Suggested script:
                         - आपके क्षेत्र ([customer_area]) में इन प्रोडक्ट्स की भी ज़्यादा मांग है:
                          [top_area_products].
                         - क्या आप इनमें से कुछ और लेना चाहेंगे?

                         Step 4: Conversation Flow Summary
                         * Customer से नाम लेकर greet करें।
                         * Customer से आज का order लें।
                         * Past order के साथ match करें:
                         - Same: confirm करें।
                         - Missing items: याद दिलाएँ और पूछें।
                         - New items: cross-sell suggestions दें।
                         * Area-wise popular products suggest करें।
                         * Natural, one-step-at-a-time conversation रखें।
                     16. किसी भी प्रोडक्ट के नाम में शॉर्ट फॉर्म या शॉर्टकट लिखा हो, तो उसे सीधे न बोलें।
                        उदाहरण के लिए:
                        - अगर लिखा है 'Haldiram Navratan (24pc)' तो '24 पीस’ न बोलें, बल्कि कहें: '24 पैकेट वाला यह पैक है।’
                        - अगर लिखा है 'Britannia Good Day - Nut Cashew' तो उसे किसी गलत या अधूरे तरीके से न बोलें (जैसे 'Nut Cashew' या 'mynuse Nut Cashew')। इसकी जगह पूरा नाम सही तरह से बोलें: 'Britannia Good Day Nut Cashew।'
                     17. लॉजिक (स्टेप-बाय-स्टेप):
                         *ग्राहक जो प्रोडक्ट ऑर्डर करे, उसकी नाम + पैक/क्वांटिटी CSV के डेटा से मैच करें।
                         *अगर नाम समान है लेकिन साइज या प्राइस अलग है, तो ग्राहक से कन्फर्म करें कि कौन सा चाहिए।
                         *ग्राहक की पुष्टि के बाद ही ऑर्डर आगे बढ़ाएँ।

                         # उदाहरण बातचीत स्टाइल (महिला टोन):
                         ग्राहक: मुझे रजनीगन्धा चाहिए।
                         आप: जी, रजनीगन्धा के कई विकल्प हैं:
                         - 100g
                         - 17g (1 Pcs)
                         - 17g IC (12 Pcs)
                         - 4g (52 Pcs)
                         - 17g Zipper IC
                         - 4g (104 Pcs)
                        आप कृपया बताएं कि आपको कौन सा पैक चाहिए?

                         ग्राहक: मुझे 52 pcs वाला चाहिए।
                         आप: ठीक है, 52 pcs वाला रजनीगन्धा ही ऑर्डर कर दूँ?

                        - Haldiram उदाहरण:
                         ग्राहक: मुझे Haldiram - Salted Peanuts चाहिए।
                         आप: जी, इसके भी कई विकल्प हैं:
                         - 5 रुपए वाला (single pcs)
                         - 24 pcs वाला (₹240)
                         आप कृपया बताएं कि आपको कौन सा पैक चाहिए?

                     18. ग्राहक का विवरण जैसे उसका नाम, फोन नंबर, पता, दुकान का नाम आदि order-detail.csv से लें और JSON बनाते समय इन विवरणों का उपयोग करें।
                     19. ग्राहक का ऑर्डर कन्फ़र्म होते ही चुपचाप JSON बना लें। फिर ग्राहक को प्रेमपूर्वक से कहें: ‘ऑर्डर करने के लिए और DS Group पर भरोसा करने के लिए आपका बहुत-बहुत धन्यवाद।’ इसके बाद कॉल को विनम्रता से समाप्त करें
                     20. अगर किसी प्रोडक्ट का नाम समान (एक जैसा) हो, तो ग्राहक को उन प्रोडक्ट्स के बारे में जानकारी दें और फिर पूछकर कन्फ़र्म करें कि उन्हें कौन-सा प्रोडक्ट चाहिए।
                     21. बार-बार पूरा प्रोडक्ट नाम लेकर न बोलें। बस ग्राहक को इस तरह बताएं कि हमारे पास इस ब्रांड में ये-ये चीज़ें भी उपलब्ध हैं। जैसे: ‘Haldiram में हमारे पास Navratan, Punjabi Tadka, Nutcracker, Salted Peanut आदि उपलब्ध हैं।’ इस प्रकार बोलकर ग्राहक से पूछें कि इनमें से कौन-सा प्रोडक्ट चाहिए।
                     22. JSON FILLING rule :
                         * Product Catalog (165 items) → इसमें से price, product name, offers (5% discount, 1 pack free) निकलें।
                         * Order Detail CSV → इसमें customer की details हैं (जैसे name, number, shop name, address आदि)।
                         * Output → एक JSON बनाना है जिसमें:
                           - Customer details order-detail.csv से आएँ।
                           - Products + Quantity customer की demand से आए।
                           - हर बार जब भी customer product quantity बोले → आप check करो कि offer applicable है या पास है → और तुरंत suggest करो।
                           - पैक आधारित ऑफर प्रत्येक उत्पाद के लिए लागू होते हैं और इस विवरण को JSON में भी जोड़ें (5 packs → 5% off, 10 packs → 1 free pack)।
                           - Cart Value Based Offers (पूरे order पर)
                              ₹25,000 की shopping → अगर customer ₹25,000 complete कर लेता है → तुरंत ₹2,000 discount।
                              ₹50,000 की shopping → अगर customer ₹50,000 complete कर लेता है → तुरंत ₹5,500 discount।
                           - JSON बनाते समय केवल एक ही प्रस्ताव अंतिम में लागू करें और ग्राहक को इसके बारे में बताएं।
                           - साथ ही suggestion message भी JSON में हो, ताकि सिस्टम या बॉट ग्राहक को friendly तरीके से बोल सके।
                           - total_amount_after_discount = total_amount - pack_based_discount - cart_value_discount
                           - Cart Value Discount Apply करो:
                            1. जब "total_amount" निकल जाए, check करो:
                              - अगर ≥ 25000 → cart_value_discount = 2000
                              - अगर ≥ 50000 → cart_value_discount = 5500
                            2. नहीं तो null.
                            - Products Loop में Discount भी निकालो
                              1. product check करो कि quantity 5 या 10 तो नहीं।
                              2. अगर 5 → 5% discount add करो।
                              3. अगर 10 → 1 free pack (PTR x 1) discount add करो।
                              4. यह discount pack_based_discount नाम के variable में जोड़ो।

                      --------------------
                      ### : Hidden backend task (ग्राहक को कभी न बताएँ):
                      कॉल समाप्त होते ही, चुपचाप JSON इस सटीक फॉर्मेट में बनाएँ:

                      {
                        "customer_details": {
                          "name": "<customer name>",
                          "store_name": "<store/shop name>",
                          "delivery_address": {
                            "house_no": "<house/building>",
                            "street": "<street/road>",
                            "area": "<area/colony>",
                            "city": "<city>",
                            "pincode": "<pincode>"
                          }
                        },
                        "order": {
                          "products": [
                            {
                              "sku": "<sku from catalog>",
                              "name": "<product name>",
                              "quantity": "<number>",
                              "unit": "<pack/carton>",
                              "packOf":"<packOf>",
                              "MRP": "<MRP>",
                              "PTR": "<PTR>",
                              "Total": "<PTR * quantity>",
                              "Offer":"<Pack Based Offers/null>"
                            }
                          ],
                          "total_products": "<total no. of products>",
                          "total_amount": "<sum of all totals>",
                          "total_amount_after_discount": "<total_amount - Pack Based Discount - Cart Value Based Discount>",
                          "cart_value_offer": "<null | '₹2,000 off on ₹25,000+' | '₹5,500 off on ₹50,000+'>",
                          "payment_method": "<CASH | UPI | CREDIT | OTHER>",
                          "confirmation": true,
                          "remarks": "<any notes>"
                        }
                      }
                      --------------------
                     ### : Catalog usage rules
                      - केवल दिए गए कैटलॉग SKUs (status = ACTIVE) का प्रयोग करें।
                      - कोई नकली SKU न बनाएँ। यदि प्रोडक्ट न मिले तो शिष्टतापूर्वक विकल्प सुझाएँ।
                      - JSON केवल बैकएंड के लिए है, कभी बोलकर न बताएँ।
"""
)

# Substitute the actual JSON text into the template
SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.substitute(
    catalog_json=catalog_json,
    orders_json=orders_json,
)

# Debugging quick-check (optional)
print("SYSTEM_PROMPT length:", len(SYSTEM_PROMPT))
print("SYSTEM_PROMPT preview:\n", SYSTEM_PROMPT[:80000])

# Rebuild CONFIG using the filled-in SYSTEM_PROMPT
CONFIG = types.LiveConnectConfig(
    system_instruction=types.Content(
        parts=[types.Part(text=SYSTEM_PROMPT)]
    ),
    generation_config=types.GenerationConfig(
        response_modalities=["AUDIO"]
    ),
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name="Aoede"
            )
        )
    ),
)

# ================== Exotel Utils ==================
def make_exotel_call(to_number: str):
    """
    Initiate an Exotel call using HTTP Basic Auth

    IMPORTANT: This connects the call to an Exotel Flow that contains a Stream/Voicebot Applet.
    You MUST create a Flow in Exotel's App Bazaar first with a Stream/Voicebot Applet configured
    to connect to your WebSocket endpoint: wss://{NGROK_URL}/ws
    """
    # API endpoint with subdomain and Account SID in path
    url = f"https://{EXOTEL_SUBDOMAIN}/v1/Accounts/{EXOTEL_ACCOUNT_SID}/Calls/connect.json"

    # IMPORTANT: Use either "To" OR "Url" parameter, NOT BOTH!
    # - Use "To" to connect to another phone number
    # - Use "Url" to connect to a Flow/Applet (for WebSocket streaming)
    payload = {
        "From": to_number,  # The customer's phone number to call first
        "CallerId": EXOTEL_SOURCE,  # Your Exotel virtual number
        "Url": f"http://my.exotel.com/{EXOTEL_ACCOUNT_SID}/exoml/start_voice/{EXOTEL_FLOW_ID}",  # Flow with Stream/Voicebot Applet
    }

    # Use API Key and API Token for authentication (NOT the Account SID)
    resp = requests.post(url, data=payload, auth=(EXOTEL_API_KEY, EXOTEL_API_TOKEN))

    # Log the response for debugging
    logging.info(f"Exotel API Response Status: {resp.status_code}")
    if resp.status_code != 200:
        logging.error(f"Exotel API Error: {resp.text}")
    else:
        logging.info(f" Call initiated successfully to {to_number}")

    return resp.json()

# ================== Audio Resampling Helper ==================
def resample_audio(audio_data: bytes, orig_rate: int = 24000, target_rate: int = 8000) -> bytes:
    """
    Resample audio from Gemini (24kHz) to Exotel format (8kHz)

    Args:
        audio_data: Raw PCM audio bytes (16-bit, mono)
        orig_rate: Original sample rate (Gemini outputs 24kHz)
        target_rate: Target sample rate (Exotel expects 8kHz)

    Returns:
        Resampled audio bytes  
    """
    try:
        # Convert bytes to numpy array (16-bit signed integers)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Calculate number of samples after resampling
        num_samples = int(len(audio_array) * target_rate / orig_rate)

        # Resample using scipy's signal.resample
        resampled_array = signal.resample(audio_array, num_samples)

        # Convert back to 16-bit integers and then to bytes
        return resampled_array.astype(np.int16).tobytes()

    except Exception as e:
        logging.error(f"❌ Error resampling audio: {e}")
        # Return original audio if resampling fails
        return audio_data

# ================== Gemini Audio Handler ==================
class AudioLoop:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.session = None
        self.running = True
        self.audio_buffer = []
        self.buffer_size = 1  # Send audio in near real-time for better responsiveness
        self.last_audio_had_sound = False

    def stop(self):
        self.running = False

    def has_audio_activity(self, audio_bytes: bytes, threshold: int = 500) -> bool:
        """Check if audio contains activity (not just silence)"""
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            # Calculate RMS (Root Mean Square) to detect audio activity
            rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float64))))
            return rms > threshold
        except Exception as e:
            logging.debug(f"Speech detection error: {e}")
            return True  # Assume activity if we can't check

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                logging.info("✅ Gemini session established - Starting audio loops")

                # Start tasks to handle bidirectional audio
                # No initial trigger - let Gemini respond naturally to incoming audio
                tg.create_task(self.receive_from_exotel())
                tg.create_task(self.send_to_exotel())
        except Exception as e:
            logging.error(f"❌ Error in AudioLoop: {e}")
            traceback.print_exc()

    async def receive_from_exotel(self):
        """Receive audio from Exotel WebSocket and send to Gemini"""
        try:
            while self.running:
                # Receive message from Exotel
                message = await self.websocket.receive_json()
                event = message.get("event")

                if event == "start":
                    logging.info("📞 Call started - WebSocket connection established")
                    logging.info("🎤 Starting to listen for customer audio...")

                elif event == "media":
                    # Get audio data from Exotel (base64-encoded PCM)
                    media_payload = message.get("media", {}).get("payload", "")

                    if media_payload:
                        # Decode base64 audio
                        audio_bytes = base64.b64decode(media_payload)

                        # Add to buffer
                        self.audio_buffer.append(audio_bytes)

                        # Send buffered audio when we have enough chunks
                        if len(self.audio_buffer) >= self.buffer_size:
                            # Combine buffered chunks
                            combined_audio = b''.join(self.audio_buffer)

                            # Check if audio contains actual sound
                            has_sound = self.has_audio_activity(combined_audio)

                            # Log speech detection
                            if has_sound and not self.last_audio_had_sound:
                                logging.info("🎤 SPEECH DETECTED in user audio!")
                                self.last_audio_had_sound = True
                            elif not has_sound and self.last_audio_had_sound:
                                logging.info("🔇 Silence detected")
                                self.last_audio_had_sound = False

                            # Send audio to Gemini using correct API signature
                            # Exotel sends 8kHz PCM audio (16-bit, mono)
                            try:
                                await self.session.send_realtime_input(
                                    audio=types.Blob(
                                        data=combined_audio,
                                        mime_type="audio/pcm;rate=8000"
                                    )
                                )
                            except Exception as e:
                                logging.error(f"❌ Failed to send audio to Gemini: {e}")

                            # Clear buffer
                            self.audio_buffer = []

                            # Log every audio send for debugging
                            if not hasattr(self, 'chunk_count'):
                                self.chunk_count = 0
                            self.chunk_count += self.buffer_size
                            if self.chunk_count % 10 == 0:
                                logging.info(f"📥 Sent {self.chunk_count} audio chunks ({len(combined_audio)} bytes) from Exotel → Gemini")

                elif event == "stop":
                    logging.info("📞 Call ended by customer")

                    # Flush any remaining buffered audio
                    if self.audio_buffer:
                        combined_audio = b''.join(self.audio_buffer)
                        await self.session.send_realtime_input(
                            audio=types.Blob(
                                data=combined_audio,
                                mime_type="audio/pcm;rate=8000"
                            )
                        )
                        logging.info(f"📤 Flushed {len(self.audio_buffer)} remaining audio chunks")
                        self.audio_buffer = []

                    self.running = False
                    break

                else:
                    logging.debug(f"🔔 Received event: {event}")

        except Exception as e:
            logging.error(f"❌ Error receiving from Exotel: {e}")
            traceback.print_exc()
            self.running = False

    async def send_to_exotel(self):
        """Receive audio from Gemini and send to Exotel WebSocket"""
        try:
            logging.info("🎧 Starting to listen for Gemini responses...")
            response_count = 0
            audio_sent_count = 0
            turn_number = 0

            # Outer loop for continuous multi-turn conversation
            while self.running:
                turn_number += 1
                logging.info(f"🔄 Starting turn #{turn_number}")

                # Inner loop receives responses for one turn
                async for response in self.session.receive():
                    if not self.running:
                        logging.info("🛑 Stopping send_to_exotel - call ended")
                        break

                    response_count += 1

                    # Log response details
                    logging.info(f"📨 Response #{response_count} from Gemini - Type: {type(response).__name__}")

                    # Heartbeat
                    if response_count % 20 == 0:
                        logging.info(f"💓 Heartbeat: Processed {response_count} responses, sent {audio_sent_count} audio chunks")

                    # Log text responses
                    if text := response.text:
                        logging.info(f"🤖 Gemini Text: {text}")

                    # Handle audio responses from Gemini
                    if hasattr(response, 'server_content') and response.server_content:
                        server_content = response.server_content

                        # Check for turn_complete to break from inner loop
                        if hasattr(server_content, 'turn_complete') and server_content.turn_complete:
                            logging.info(f"✅ Turn #{turn_number} complete - Ready for next turn")
                            break  # Exit inner loop only, continue outer loop for next turn

                        # Check for inline_data directly in server_content
                        if hasattr(server_content, 'inline_data') and server_content.inline_data:
                            audio_data = server_content.inline_data.data

                            # Resample from Gemini's 24kHz to Exotel's 8kHz
                            resampled_audio = resample_audio(audio_data)

                            # Encode to base64 for Exotel
                            audio_base64 = base64.b64encode(resampled_audio).decode('utf-8')

                            # Send audio back to Exotel
                            await self.websocket.send_json({
                                "event": "media",
                                "media": {
                                    "payload": audio_base64
                                }
                            })
                            audio_sent_count += 1
                            logging.info(f"📤 Sent {len(resampled_audio)} bytes of audio to Exotel (resampled from {len(audio_data)} bytes)")

                        # Check for model_turn with parts
                        elif hasattr(server_content, 'model_turn') and server_content.model_turn:
                            model_turn = server_content.model_turn
                            if hasattr(model_turn, 'parts'):
                                for part in model_turn.parts:
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        audio_data = part.inline_data.data

                                        # Resample from Gemini's 24kHz to Exotel's 8kHz
                                        resampled_audio = resample_audio(audio_data)

                                        # Encode to base64 for Exotel
                                        audio_base64 = base64.b64encode(resampled_audio).decode('utf-8')

                                        # Send audio back to Exotel
                                        await self.websocket.send_json({
                                            "event": "media",
                                            "media": {
                                                "payload": audio_base64
                                            }
                                        })
                                        audio_sent_count += 1
                                        logging.info(f"📤 Sent {len(resampled_audio)} bytes of audio to Exotel (resampled from {len(audio_data)} bytes)")

                    # Also check for data attribute directly on response
                    elif hasattr(response, 'data') and response.data:
                        audio_data = response.data

                        # Resample from Gemini's 24kHz to Exotel's 8kHz
                        resampled_audio = resample_audio(audio_data)

                        # Encode to base64 for Exotel
                        audio_base64 = base64.b64encode(resampled_audio).decode('utf-8')

                        # Send audio back to Exotel
                        await self.websocket.send_json({
                            "event": "media",
                            "media": {
                                "payload": audio_base64
                            }
                        })
                        audio_sent_count += 1
                        logging.info(f"📤 Sent {len(resampled_audio)} bytes of audio to Exotel (resampled from {len(audio_data)} bytes)")

                # If inner loop exits normally (not via break), log it
                logging.info(f"📭 Turn #{turn_number} receive loop completed")

            logging.info("🛑 Outer conversation loop ended")

        except Exception as e:
            logging.error(f"❌ Error sending to Exotel: {e}")
            traceback.print_exc()
            self.running = False
        finally:
            logging.warning("⚠️ send_to_exotel method ended")

# ================== WebSocket Bridge ==================
@app.websocket("/ws")
async def websocket_bridge(ws: WebSocket):
    await ws.accept()
    logging.info("📞 Exotel stream connected")

    ai_loop = AudioLoop(websocket=ws)

    try:
        await ai_loop.run()
    except Exception as e:
        logging.error(f"❌ WebSocket error: {e}")
        traceback.print_exc()
    finally:
        logging.info("📞 WebSocket connection closed")

# ================== Exotel Call Endpoint ==================
class CallRequest(BaseModel):
    to: str

@app.post("/make-call")
async def make_call(req: CallRequest):
    response = make_exotel_call(req.to)
    logging.info(f"📞 Exotel call initiated to {req.to}")
    return response

# ================== Utility: Free Port Finder ==================
def get_free_port(default_port=8000):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("", default_port))
        port = s.getsockname()[1]
    except OSError:
        s.bind(("", 0))
        port = s.getsockname()[1]
    finally:
        s.close()
    return port

# ================== Streamlit UI ==================
def streamlit_ui():
    st.title("📞 Kredmint Voice Assistant")
    st.write("Press the button to call and let the AI greet automatically via Exotel.")

    to_number = st.text_input("Enter phone number (with country code)", "+91XXXXXXXXXX")

    if st.button("📲 Call Now"):
        if to_number:
            res = make_exotel_call(to_number)
            st.success("✅ Call initiated successfully!")
            st.json(res)
        else:
            st.warning("Please enter a valid number!")

# ================== Entrypoint ==================
if __name__ == "__main__":

    def run_fastapi():
        default_port = int(os.getenv("PORT", 8501))
        port = get_free_port(default_port)
        print(f"🚀 Starting FastAPI on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)

    # Prevent multiple FastAPI restarts in Streamlit reruns
    if not any(t.name == "FastAPIThread" for t in threading.enumerate()):
        threading.Thread(target=run_fastapi, name="FastAPIThread", daemon=True).start()

    # Run Streamlit UI
    streamlit_ui()
