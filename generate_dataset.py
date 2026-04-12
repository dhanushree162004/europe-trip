import json
import random

countries = ["France", "Italy", "Germany", "Spain", "Netherlands", "Switzerland", "Norway"]
categories = ["Attraction", "Cost", "Transport", "Insight", "FAQ", "Comparison", "Multilingual"]

documents = []
doc_id = 1

# 1. Places/Attractions with City (approx 70 docs)
attractions_data = {
    "France": [
        ("Eiffel Tower", "Paris", 25, "luxury", "Iconic landmark with breathtaking views."),
        ("Louvre Museum", "Paris", 17, "cultural", "World's largest art museum."),
        ("Mont Saint-Michel", "Normandy", 11, "cultural", "Stunning island commune."),
        ("Palace of Versailles", "Versailles", 20, "luxury", "Gilded former royal residence."),
        ("Cote d'Azur Beaches", "Nice", 0, "budget", "Beautiful public beaches along the Riviera."),
        ("Sainte-Chapelle", "Paris", 11, "cultural", "Stunning gothic chapel with stained glass."),
        ("Musee d'Orsay", "Paris", 16, "cultural", "Impressionist masterpieces."),
        ("Arc de Triomphe", "Paris", 13, "cultural", "Historic monument at Champs-Élysées."),
        ("Seine River Cruise", "Paris", 15, "luxury", "Relaxing boat tour through Paris."),
        ("Verdon Gorge", "Provence", 0, "adventure", "Deep river canyon for hiking."),
        ("Lyon Old Town", "Lyon", 0, "cultural", "Medieval and Renaissance architecture."),
        ("Mont Blanc", "Chamonix", 50, "adventure", "Highest mountain in the Alps.")
    ],
    "Italy": [
        ("Colosseum", "Rome", 16, "cultural", "Epic ancient amphitheater."),
        ("Pantheon", "Rome", 0, "cultural", "Best-preserved ancient Roman temple."),
        ("Venice Canals", "Venice", 80, "luxury", "Romantic gondola rides."),
        ("Duomo di Milano", "Milan", 15, "cultural", "Stunning gothic cathedral."),
        ("Cinque Terre Hike", "Cinque Terre", 8, "adventure", "Scenic coastal trails."),
        ("Uffizi Gallery", "Florence", 20, "cultural", "Renaissance art masterpieces."),
        ("Pompeii Ruins", "Naples", 18, "cultural", "Ancient Roman city ruins."),
        ("Amalfi Coast", "Amalfi", 0, "luxury", "Dramatic coastal drive."),
        ("Lake Como", "Como", 0, "luxury", "Sophisticated lake retreat."),
        ("Vatican Museums", "Vatican City", 17, "cultural", "Sistine Chapel and more."),
        ("Trevi Fountain", "Rome", 0, "budget", "Magical Baroque fountain."),
        ("Leaning Tower", "Pisa", 18, "cultural", "Famous tilted bell tower.")
    ],
    "Germany": [
        ("Neuschwanstein Castle", "Fussen", 15, "luxury", "Fairytale castle."),
        ("Berlin Wall Memorial", "Berlin", 0, "cultural", "History of divided Germany."),
        ("Brandenburg Gate", "Berlin", 0, "cultural", "Symbol of German reunification."),
        ("Black Forest", "Baden-Baden", 0, "adventure", "Dense woods and villages."),
        ("Oktoberfest Munich", "Munich", 100, "luxury", "World-renowned beer festival."),
        ("Miniatur Wunderland", "Hamburg", 20, "budget", "Largest model railway."),
        ("Cologne Cathedral", "Cologne", 0, "cultural", "Majestic gothic cathedral."),
        ("Europa-Park", "Rust", 55, "adventure", "Huge theme park."),
        ("Marienplatz", "Munich", 0, "cultural", "Central square with Glockenspiel."),
        ("Saxon Switzerland", "Dresden", 0, "adventure", "Sandstone formations for hiking."),
        ("Rhine Valley Cruise", "Koblenz", 25, "cultural", "Castles and vineyards."),
        ("Checkpoint Charlie", "Berlin", 0, "cultural", "Historic border crossing.")
    ],
    "Spain": [
        ("Sagrada Familia", "Barcelona", 26, "cultural", "Gaudi's masterpiece."),
        ("Park Guell", "Barcelona", 10, "budget", "Playful mosaic park."),
        ("Alhambra", "Granada", 14, "cultural", "Moorish palace and fortress."),
        ("Prado Museum", "Madrid", 15, "cultural", "Main national art museum."),
        ("La Tomatina", "Valencia", 12, "adventure", "Tomato throwing festival."),
        ("Ibiza Nightlife", "Ibiza", 50, "luxury", "Premier music clubs."),
        ("Seville Cathedral", "Seville", 12, "cultural", "Largest gothic cathedral."),
        ("Running of the Bulls", "Pamplona", 0, "adventure", "Historic tradition."),
        ("Costa del Sol", "Malaga", 0, "budget", "Sunny beaches."),
        ("Royal Palace of Madrid", "Madrid", 13, "luxury", "Official royal residence."),
        ("Basilica del Pilar", "Zaragoza", 0, "cultural", "Grand baroque cathedral."),
        ("Montserrat Monastery", "Barcelona", 10, "adventure", "Mountain retreat.")
    ],
    "Netherlands": [
        ("Rijksmuseum", "Amsterdam", 20, "cultural", "Dutch national museum."),
        ("Anne Frank House", "Amsterdam", 16, "cultural", "Hiding place museum."),
        ("Keukenhof Gardens", "Lisse", 19, "luxury", "Tulip park."),
        ("Kinderdijk Windmills", "Rotterdam", 16, "cultural", "Historic windmills."),
        ("Canal Cruise Amsterdam", "Amsterdam", 15, "budget", "Historic city from water."),
        ("Van Gogh Museum", "Amsterdam", 20, "cultural", "Vincent van Gogh works."),
        ("Efteling", "Kaatsheuvel", 45, "adventure", "Fairytale park."),
        ("Madurodam", "The Hague", 18, "budget", "Miniature park."),
        ("Zaanse Schans", "Zaandam", 0, "cultural", "Historic windmills."),
        ("Giethoorn Village", "Giethoorn", 0, "budget", "Car-free village."),
        ("Hoge Veluwe", "Otterlo", 11, "adventure", "Forest with bikes."),
        ("Peace Palace", "The Hague", 10, "cultural", "International Court.")
    ],
    "Switzerland": [
        ("Jungfraujoch", "Interlaken", 200, "luxury", "Top of Europe railway."),
        ("Mount Pilatus", "Lucerne", 72, "adventure", "Cogwheel railway."),
        ("Lake Geneva", "Geneva", 0, "luxury", "Sophisticated lakefront."),
        ("Chapel Bridge", "Lucerne", 0, "cultural", "Historic wooden bridge."),
        ("Matterhorn Zermatt", "Zermatt", 100, "adventure", "Iconic mountain."),
        ("Rhine Falls", "Schaffhausen", 5, "budget", "Powerful waterfall."),
        ("Interlaken", "Interlaken", 0, "adventure", "Paragliding hub."),
        ("St. Moritz", "St. Moritz", 150, "luxury", "Alpine resort town."),
        ("Swiss National Museum", "Zurich", 10, "cultural", "History of Switzerland."),
        ("Chillon Castle", "Montreux", 15, "cultural", "Medieval island castle."),
        ("Bern Old City", "Bern", 0, "cultural", "UNESCO medieval fountains."),
        ("Mount Titlis", "Engelberg", 92, "adventure", "Rotating cable car.")
    ],
    "Norway": [
        ("Nærøyfjord", "Gudvangen", 45, "adventure", "Stunning UNESCO fjord cruise."),
        ("Vigeland Park", "Oslo", 0, "cultural", "World's largest sculpture park."),
        ("Flåm Railway", "Flåm", 65, "luxury", "Incredible scenic mountain rail."),
        ("Munch Museum", "Oslo", 18, "cultural", "Art museum dedicated to Edvard Munch."),
        ("Bryggen", "Bergen", 0, "cultural", "Historic Hanseatic wharf."),
        ("Preikestolen", "Stavanger", 0, "adventure", "Famous 'Pulpit Rock' cliff."),
        ("Oslo Opera House", "Oslo", 10, "cultural", "Modern marble architectural marvel guided tour.")
    ],
    "Sweden": [
        ("Vasa Museum", "Stockholm", 18, "cultural", "17th-century warship museum."),
        ("Gamla Stan", "Stockholm", 0, "cultural", "Stockholm's historic old town."),
        ("Skansen", "Stockholm", 22, "family", "World's oldest open-air museum."),
        ("ABBA The Museum", "Stockholm", 25, "cultural", "Interactive pop music museum."),
        ("Stockholm Archipelago", "Stockholm", 35, "adventure", "Boat tour through 30,000 islands.")
    ]
}

for country, items in attractions_data.items():
    for name, city, cost, pref, desc in items:
        documents.append({
            "id": doc_id,
            "country": country,
            "category": "Attraction",
            "content": f"{name}: {desc} Located in {city}, {country}.",
            "metadata": {"cost": cost, "duration": 3, "type": pref, "name": name, "city": city}
        })
        doc_id += 1

# 2. Cost Knowledge (Dining & City base)
city_costs = [
    ("Paris", "Boulangerie breakfast costs around €8; Casual bistro lunch €20; Seated dinner €40."),
    ("Rome", "Trattoria pasta bowl €12-15; Gelato €3-5; Pizza al taglio €5."),
    ("Barcelona", "Tapas dinner €15-20; Set lunch (Menu del Dia) €12-15; Coffee €2."),
    ("Berlin", "Currywurst €5; Döner kebab €7; Beer in pub €4."),
    ("Amsterdam", "Pancakes €12; Herring snack €5; Indonesian Rijsttafel €30."),
    ("Zurich", "Basic lunch €25; Cheese fondue €35; Coffee €6."),
    ("Madrid", "Churros con chocolate €5; Calamari sandwich €8; Dinner at Mercado de San Miguel €25."),
    ("Munich", "Pretzel €2; Beer garden meal €15; Schnitzel €18."),
    ("Geneva", "Swiss chocolate box €15; Lakefront lunch €30; Dinner €50."),
    ("Oslo", "Salmon dish €30; Local brown cheese snack €5; Coffee €7."),
    ("Bergen", "Fish soup €18; Reindeer burger €25; Local cider €12.")
]

for city, content in city_costs:
    documents.append({
        "id": doc_id,
        "country": "Multi",
        "category": "Cost",
        "content": f"Dining in {city}: {content}",
        "metadata": {"type": "dining", "city": city}
    })
    doc_id += 1

# 3. Transport Knowledge (City to City)
intercity_transport = [
    ("Barcelona to Madrid", "AVE High-speed train: €40-90, Duration: 2.5 hours."),
    ("Paris to Lyon", "TGV train: €30-70, Duration: 2 hours."),
    ("Rome to Florence", "Trenitalia: €20-50, Duration: 1.5 hours."),
    ("Amsterdam to Rotterdam", "Intercity train: €15, Duration: 40 minutes."),
    ("Berlin to Munich", "ICE train: €45-120, Duration: 4 hours."),
    ("Zurich to Geneva", "SBB train: €50-90, Duration: 2.8 hours."),
    ("Paris to Versailles", "RER C train: €4, Duration: 45 minutes."),
    ("Madrid to Seville", "AVE train: €35-75, Duration: 2.5 hours."),
    ("Milan to Venice", "Frecciarossa train: €25-60, Duration: 2.2 hours."),
    ("Amsterdam to The Hague", "NS train: €12, Duration: 50 minutes."),
    ("Oslo to Bergen", "Bergensbanen: €70-120, Duration: 7 hours.")
]

for route, details in intercity_transport:
    documents.append({
        "id": doc_id,
        "country": "Multi",
        "category": "Transport",
        "content": f"Transport from {route}: {details}",
        "metadata": {"type": "intercity", "route": route}
    })
    doc_id += 1

# 4. Hotel Dataset (v14.0)
hotels_data = [
    # FRANCE
    {"city": "Paris", "country": "France", "hotel_name": "ibis Paris Centre", "type": "solo", "price_range": "budget", "avg_price": 85, "features": ["central", "wifi"], "rating": 4.1},
    {"city": "Paris", "country": "France", "hotel_name": "Hotel de l'Avenir", "type": "family", "price_range": "budget", "avg_price": 95, "features": ["family-rooms", "metro"], "rating": 4.2},
    {"city": "Paris", "country": "France", "hotel_name": "Novotel Paris Les Halles", "type": "family", "price_range": "mid", "avg_price": 220, "features": ["spacious", "central"], "rating": 4.5},
    {"city": "Paris", "country": "France", "hotel_name": "Pullman Paris Tour Eiffel", "type": "couple", "price_range": "mid", "avg_price": 280, "features": ["view", "modern"], "rating": 4.6},
    {"city": "Paris", "country": "France", "hotel_name": "Shangri-La Paris", "type": "couple", "price_range": "luxury", "avg_price": 1200, "features": ["palace", "luxury"], "rating": 4.9},
    {"city": "Paris", "country": "France", "hotel_name": "The Ritz Paris", "type": "group", "price_range": "luxury", "avg_price": 1500, "features": ["iconic", "service"], "rating": 5.0},
    
    # ITALY
    {"city": "Rome", "country": "Italy", "hotel_name": "Hotel Trastevere", "type": "solo", "price_range": "budget", "avg_price": 70, "features": ["vibrant", "authentic"], "rating": 4.3},
    {"city": "Rome", "country": "Italy", "hotel_name": "Generator Rome", "type": "group", "price_range": "budget", "avg_price": 60, "features": ["social", "modern"], "rating": 4.0},
    {"city": "Rome", "country": "Italy", "hotel_name": "Hotel Indigo Rome", "type": "couple", "price_range": "mid", "avg_price": 240, "features": ["design", "boutique"], "rating": 4.7},
    {"city": "Rome", "country": "Italy", "hotel_name": "Hotel Quirinale", "type": "family", "price_range": "mid", "avg_price": 190, "features": ["classic", "theatre"], "rating": 4.4},
    {"city": "Rome", "country": "Italy", "hotel_name": "Hotel Hassler", "type": "luxury", "price_range": "luxury", "avg_price": 850, "features": ["spanish-steps", "exclusive"], "rating": 4.9},

    # NETHERLANDS
    {"city": "Amsterdam", "country": "Netherlands", "hotel_name": "Hans Brinker Hostel", "type": "group", "price_range": "budget", "avg_price": 45, "features": ["central", "party"], "rating": 3.8},
    {"city": "Amsterdam", "country": "Netherlands", "hotel_name": "CityHub Amsterdam", "type": "solo", "price_range": "budget", "avg_price": 80, "features": ["tech", "unique"], "rating": 4.4},
    {"city": "Amsterdam", "country": "Netherlands", "hotel_name": "CitizenM Amstel", "type": "couple", "price_range": "mid", "avg_price": 180, "features": ["modern", "smart"], "rating": 4.6},
    {"city": "Amsterdam", "country": "Netherlands", "hotel_name": "Pulitzer Amsterdam", "type": "family", "price_range": "luxury", "avg_price": 600, "features": ["canal-house", "historic"], "rating": 4.8},

    # GERMANY
    {"city": "Berlin", "country": "Germany", "hotel_name": "Meininger Berlin", "type": "family", "price_range": "budget", "avg_price": 65, "features": ["kitchen", "family-friendly"], "rating": 4.1},
    {"city": "Berlin", "country": "Germany", "hotel_name": "25hours Hotel Bikini", "type": "couple", "price_range": "mid", "avg_price": 170, "features": ["zoo-view", "quirky"], "rating": 4.6},
    {"city": "Berlin", "country": "Germany", "hotel_name": "Hotel Adlon Kempinski", "type": "luxury", "price_range": "luxury", "avg_price": 550, "features": ["historic", "grand"], "rating": 4.9},

    # SWITZERLAND
    {"city": "Zurich", "country": "Switzerland", "hotel_name": "Oldtown Hostel Otter", "type": "solo", "price_range": "budget", "avg_price": 90, "features": ["central", "bar"], "rating": 4.2},
    {"city": "Zurich", "country": "Switzerland", "hotel_name": "25hours Hotel Langstrasse", "type": "couple", "price_range": "mid", "avg_price": 280, "features": ["vibrant", "urban"], "rating": 4.5},
    {"city": "Zurich", "country": "Switzerland", "hotel_name": "The Dolder Grand", "type": "luxury", "price_range": "luxury", "avg_price": 900, "features": ["castle", "spa"], "rating": 4.9},
    
    # NORWAY
    {"city": "Oslo", "country": "Norway", "hotel_name": "Comfort Hotel Xpress", "type": "solo", "price_range": "budget", "avg_price": 95, "features": ["central", "urban"], "rating": 4.1},
    {"city": "Oslo", "country": "Norway", "hotel_name": "The Thief", "type": "couple", "price_range": "luxury", "avg_price": 450, "features": ["art", "waterfront"], "rating": 4.8},
    {"city": "Oslo", "country": "Norway", "hotel_name": "Clarion Hotel The Hub", "type": "family", "price_range": "mid", "avg_price": 220, "features": ["pool", "rooftop"], "rating": 4.4},
    {"city": "Gudvangen", "country": "Norway", "hotel_name": "Gudvangen Fjordtell", "type": "couple", "price_range": "mid", "avg_price": 180, "features": ["fjord-view", "viking-theme"], "rating": 4.5},
    
    # SWEDEN
    {"city": "Stockholm", "country": "Sweden", "hotel_name": "Generator Stockholm", "type": "group", "price_range": "budget", "avg_price": 55, "features": ["central", "social"], "rating": 4.2},
    {"city": "Stockholm", "country": "Sweden", "hotel_name": "Berns Hotel", "type": "couple", "price_range": "luxury", "avg_price": 380, "features": ["historic", "nightlife"], "rating": 4.7},
    {"city": "Stockholm", "country": "Sweden", "hotel_name": "Hilton Slussen", "type": "family", "price_range": "mid", "avg_price": 250, "features": ["view", "central"], "rating": 4.5}
]

for h in hotels_data:
    documents.append({
        "id": doc_id,
        "country": h["country"],
        "category": "Hotel",
        "content": f"Hotel {h['hotel_name']} in {h['city']}: A {h['price_range']} stay for {h['type']}. Features: {', '.join(h['features'])}.",
        "metadata": {
            "name": h["hotel_name"],
            "city": h["city"],
            "type": h["type"],
            "price_range": h["price_range"],
            "avg_price": h["avg_price"],
            "rating": h["rating"]
        }
    })
    doc_id += 1

# 5. Intercity Travel Routes (v14.0)
routes_data = [
    {"from": "Paris", "to": "Amsterdam", "mode": "train", "cost": 40, "hours": 3.5, "type": "mid", "freq": "high"},
    {"from": "Paris", "to": "Amsterdam", "mode": "flight", "cost": 85, "hours": 1.2, "type": "fast", "freq": "high"},
    {"from": "Rome", "to": "Milan", "mode": "train", "cost": 45, "hours": 3.0, "type": "mid", "freq": "high"},
    {"from": "Rome", "to": "Milan", "mode": "flight", "cost": 75, "hours": 1.0, "type": "fast", "freq": "medium"},
    {"from": "Berlin", "to": "Munich", "mode": "train", "cost": 35, "hours": 4.0, "type": "mid", "freq": "high"},
    {"from": "Madrid", "to": "Barcelona", "mode": "train", "cost": 30, "hours": 2.5, "type": "fast", "freq": "high"},
    {"from": "Zurich", "to": "Geneva", "mode": "train", "cost": 60, "hours": 2.8, "type": "mid", "freq": "high"},
    
    # NORWAY ROUTES
    {"from": "Oslo", "to": "Gudvangen", "mode": "train", "cost": 30, "hours": 3.0, "type": "mid", "freq": "medium"},
    {"from": "Oslo", "to": "Bergen", "mode": "train", "cost": 45, "hours": 6.5, "type": "mid", "freq": "medium"},
    {"from": "Stockholm", "to": "Oslo", "mode": "train", "cost": 60, "hours": 5.5, "type": "mid", "freq": "low"},
    {"from": "Stockholm", "to": "Copenhagen", "mode": "train", "cost": 50, "hours": 5.0, "type": "mid", "freq": "medium"}
]

for r in routes_data:
    documents.append({
        "id": doc_id,
        "country": "Multi",
        "category": "Route",
        "content": f"Travel from {r['from']} to {r['to']} via {r['mode']}. Cost: €{r['cost']}, Duration: {r['hours']}h.",
        "metadata": {
            "from": r["from"],
            "to": r["to"],
            "mode": r["mode"],
            "cost": r["cost"],
            "hours": r["hours"],
            "type": r["type"]
        }
    })
    doc_id += 1

# Writing to file
with open("data/dataset.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=4, ensure_ascii=False)

print(f"Generated {len(documents)} logic-ready documents.")
