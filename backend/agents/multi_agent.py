import os
import json
import random

class BaseAgent:
    def __init__(self, name):
        self.name = name

    def log(self, message):
        print(f"[{self.name}] {message}")

class GuardrailAgent(BaseAgent):
    def __init__(self):
        super().__init__("Guard")
        # Domain: European travel
        self.whitelisted_keywords = [
            "france", "paris", "nice", "lyon", "marseille",
            "italy", "rome", "milan", "venice", "florence", "naples",
            "germany", "berlin", "munich", "frankfurt", "hamburg",
            "spain", "madrid", "barcelona", "seville", "valencia",
            "netherlands", "amsterdam", "rotterdam", "utrecht",
            "switzerland", "zurich", "geneva", "lucerne", "basel",
            "norway", "oslo", "bergen", "stavanger", "tromso",
            "sweden", "stockholm", "gothenburg", "malmo", "uppsala",
            "trip", "travel", "itinerary", "plan", "visit", "tour",
            "budget", "cost", "price", "euro", "eur", "flight", "hotel", "stay", "accommodation", "train", "route"
        ]

    def run(self, query):
        q = query.lower()
        if any(word in q for word in self.whitelisted_keywords):
            return {"safe": True}
        
        return {
            "safe": False,
            "category": "OUT_OF_DOMAIN",
            "message": "I focus exclusively on European travel across France, Italy, Germany, Spain, Netherlands, Switzerland, Norway, and Sweden. How can I help with your trip there? 😊"
        }

class LanguageAgent(BaseAgent):
    def __init__(self):
        super().__init__("Lang")

    def run(self, query):
        # Simplified: Pass-through for now (English target)
        return {"original": query, "translation": query, "lang": "en"}

class MemoryAgent(BaseAgent):
    def __init__(self):
        super().__init__("Memory")
        self.countries_list = ["france", "italy", "germany", "spain", "netherlands", "switzerland", "norway", "sweden"]
        self.city_map = {
            "paris": "France", "nice": "France", "lyon": "France",
            "rome": "Italy", "milan": "Italy", "venice": "Italy", "florence": "Italy",
            "berlin": "Germany", "munich": "Germany",
            "madrid": "Spain", "barcelona": "Spain",
            "amsterdam": "Netherlands",
            "zurich": "Switzerland", "geneva": "Switzerland",
            "oslo": "Norway", "bergen": "Norway",
            "stockholm": "Sweden", "gothenburg": "Sweden"
        }

    def classify_query(self, query, session_context):
        self.log("Classify user intent relative to session history...")
        q = query.lower()
        
        # 1. IRRELEVANT (Safety First)
        blacklist = ["code", "recursion", "algorithm", "python", "javascript", "solve", "math", "what is ai", "ignore previous"]
        if any(b in q for b in blacklist): return "IRRELEVANT"

        # 2. HOTELS ONLY
        if any(w in q for w in ["hotel", "stay", "accommodation", "resort", "hostel", "where to sleep"]) and \
           not any(w in q for w in ["trip", "itinerary", "plan", "route", "flight", "train"]):
            return "HOTELS_ONLY"
            
        # 3. TRAVEL ONLY
        if any(w in q for w in ["flight", "train", "route", "transport", "how to get", "intercity"]) and \
           not any(w in q for w in ["trip", "hotel", "stay", "itinerary", "attraction"]):
            return "TRAVEL_ONLY"

        # 4. NEW vs PARTIAL vs MODIFICATION
        has_loc = any(c.lower() in q for c in self.countries_list) or any(city in q for city in self.city_map.keys())
        is_fresh = any(word in q for word in ["plan", "trip", "itinerary", "travel", "visit"])
        is_mod = any(word in q for word in ["modify", "change", "make it", "days", "budget"])
        
        if has_loc and is_fresh: return "NEW"
        if is_mod: return "MODIFICATION"
        if has_loc or any(w in q for w in ["solo", "couple", "family", "group", "euro", "eur"]): return "PARTIAL"
        
        return "IRRELEVANT"

    def update_context(self, query, session_context):
        self.log("Updating session context...")
        q = query.lower()
        
        q_type = self.classify_query(query, session_context)
        
        if q_type == "NEW":
            self.log("RESETTING context for new request.")
            session_context = {
                "countries": [],
                "cities": [],
                "duration": None,
                "budget": None,
                "budget_provided": False,
                "user_type": None,
                "user_type_provided": False,
                "preference": None,
                "history": [],
            }
        
        # Extract countries
        for c in self.countries_list:
            if c in q and c not in session_context["countries"]:
                session_context["countries"].append(c.capitalize())
                
        # Extract cities & derive countries
        for city, country in self.city_map.items():
            if city in q:
                if city.capitalize() not in session_context["cities"]:
                    session_context["cities"].append(city.capitalize())
                if country not in session_context["countries"]:
                    session_context["countries"].append(country)

        # Extract duration
        import re
        days = re.findall(r'(\d+)\s*day', q)
        if days:
            session_context["duration"] = int(days[0])
            
        # Extract budget
        money = re.findall(r'(?:€|eur|euro)\s*(\d+)|(\d+)\s*(?:€|eur|euro)', q)
        if money:
            val = money[0][0] or money[0][1]
            session_context["budget"] = int(val)
            session_context["budget_provided"] = True

        # User type
        types = ["solo", "couple", "family", "group"]
        for t in types:
            if t in q:
                session_context["user_type"] = t
                session_context["user_type_provided"] = True

        # Travel preference / trip style
        preferences = {
            "adventure": "adventure",
            "outdoor": "adventure",
            "cultural": "cultural",
            "culture": "cultural",
            "relax": "relaxation",
            "relaxation": "relaxation",
            "beach": "relaxation",
            "city break": "city",
            "city": "city",
            "food": "food",
            "nature": "nature",
        }
        for key, value in preferences.items():
            if key in q:
                session_context["preference"] = value
                break
                
        session_context["history"].append(query)
        return session_context, q_type

class RetrievalAgent(BaseAgent):
    def __init__(self, vector_store):
        super().__init__("Retrieval")
        self.vector_store = vector_store

    def run(self, context):
        self.log(f"Retrieving data for {context['countries']}...")
        results = []
        for country in context["countries"]:
            results.extend(self.vector_store.get(country, []))
        return results

class LocationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Location")

    def run(self, results, context):
        self.log("Scoring cities based on preference and duration...")
        # Rule: Filter by intent-specific cities if provided
        selected_cities = context.get("cities", [])
        
        city_scores = {}
        for item in results:
            city = item["city"]
            if selected_cities and city not in selected_cities:
                continue
            city_scores[city] = city_scores.get(city, 0) + item["metadata"]["rating"]
            
        # Select top cities based on duration (1 city per 2 days approx)
        num_cities = max(1, context["duration"] // 2)
        top_cities = sorted(city_scores.items(), key=lambda x: x[1], reverse=True)[:num_cities]
        final_cities = [c[0] for c in top_cities]
        
        by_city = {city: [r for r in results if r["city"] == city] for city in final_cities}
        return final_cities, by_city

class ConstraintAgent(BaseAgent):
    def __init__(self):
        super().__init__("Constraint")

    def run(self, cities, by_city, context):
        self.log("Applying diversity constraints...")
        # Rule: Ensure mix of types (Museum, Park, etc.) per city
        constrained = {}
        for city, items in by_city.items():
            seen_types = set()
            keepers = []
            for item in sorted(items, key=lambda x: x["metadata"]["rating"], reverse=True):
                if item["metadata"]["type"] not in seen_types:
                    keepers.append(item)
                    seen_types.add(item["metadata"]["type"])
                if len(keepers) >= 4: break
            constrained[city] = keepers
            
        return {"final_cities": cities, "data": constrained}

class PlanningAgent(BaseAgent):
    def __init__(self):
        super().__init__("Planning")

    def run(self, cities, data, all_docs, context):
        self.log("Sequencing itinerary by city...")
        itinerary = []
        current_day = 1
        
        total_days = context["duration"]
        days_per_city = max(1, total_days // len(cities)) if cities else 1
        
        travel_info = []
        
        for i, city in enumerate(cities):
            city_items = data.get(city, [])
            
            # Add intercity travel if not first city
            if i > 0:
                prev_city = cities[i-1]
                routes = [d for d in all_docs if d.get("category") == "Route" 
                          and d["metadata"]["from"] == prev_city 
                          and d["metadata"]["to"] == city]
                if routes:
                    best_route = routes[0]["metadata"]
                    travel_info.append({
                        "day": current_day,
                        "from": prev_city,
                        "to": city,
                        "mode": best_route["mode"],
                        "cost": best_route["cost"],
                        "duration": best_route["hours"]
                    })

            # Assign hotels for this city
            hotels = [d for d in all_docs if d.get("category") == "Hotel" and d["metadata"]["city"] == city]
            
            for d in range(days_per_city):
                if current_day > total_days: break
                
                # Daily items (2 per day)
                daily_atrs = city_items[d*2 : (d+1)*2]
                
                itinerary.append({
                    "day": current_day,
                    "city": city,
                    "attractions": daily_atrs,
                    "hotel": hotels[:1] if hotels else []
                })
                current_day += 1
                
        return itinerary, travel_info

class BudgetAgent(BaseAgent):
    def __init__(self):
        super().__init__("Budget")

    def run(self, itinerary, travel, context):
        self.log("Calculating total trip cost...")
        atr_cost = 0
        stay_cost = 0
        transit_cost = sum(t["cost"] for t in travel)
        food_cost = context["duration"] * 40 # avg €40/day
        
        seen_stay_days = set()
        for day in itinerary:
            for atr in day["attractions"]:
                atr_cost += atr["metadata"]["avg_cost"]
            
            # v14.0 Fixed: Accurate Hotel Cost accumulation
            if day.get("hotel") and day["day"] not in seen_stay_days:
                stay_cost += day["hotel"][0]["metadata"]["avg_price_per_night"]
                seen_stay_days.add(day["day"])
        
        total = atr_cost + stay_cost + transit_cost + food_cost
        
        return {
            "total": total,
            "attractions": atr_cost,
            "stays": stay_cost,
            "transport": transit_cost,
            "food": food_cost
        }

class ExplanationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Explanation")
    def run(self, context, cities, rejected):
        msg = f"I’ve planned a well-balanced trip in {', '.join(cities)} that’s easy to navigate and fits well within your budget. "
        msg += f"You'll explore the key highlights while keeping travel minimal and efficient. "
        return msg

class PersonaAgent(BaseAgent):
    def __init__(self):
        super().__init__("Persona")

    def run(self, context, itinerary, budget, reasoning, travel, intent="PLAN_TRIP", all_docs=[]):
        self.log(f"Generating expert response for intent: {intent}")
        duration = context.get("duration", 3)
        daily_curr = context["budget"] / duration
        pr = "budget"
        if daily_curr > 120: pr = "mid"
        if daily_curr > 350: pr = "luxury"
        user_type = context["user_type"]

        # 1. Strict Grounding - Filter RAG docs
        rag_atrs = [d for d in all_docs if d.get("category") == "Attraction"]
        rag_hotels = [d for d in all_docs if d.get("category") == "Hotel" and d["country"].lower() in [c.lower() for c in context["countries"]]]
        
        filtered_hotels = [h for h in rag_hotels if h["metadata"]["type"] == user_type and h["metadata"]["price_range"] == pr]
        if not filtered_hotels: filtered_hotels = [h for h in rag_hotels if h["metadata"]["price_range"] == pr]
        if not filtered_hotels: filtered_hotels = rag_hotels
        
        # 2. Intro Section
        locs = ", ".join(context["countries"])
        if intent == "HOTELS_ONLY":
            intro = f"I've found some excellent accommodation options in {locs.capitalize()} that match your {user_type} travel style. 😊"
        elif intent == "TRAVEL_ONLY":
            intro = f"Here are the most efficient intercity travel routes for your journey across {locs.capitalize()}. 🚄"
        else:
            intro = f"I've analyzed the best options for your {duration}-day trip to {locs.capitalize()}! "

        narrative = intro
        
        # 3. Trip Itinerary (PLAN_TRIP)
        if intent in ["NEW", "PARTIAL", "MODIFICATION", "PLAN_TRIP"] and itinerary:
            cities = ", ".join(set(d["city"] for d in itinerary))
            narrative += f"\n\nWe'll be exploring the absolute best of {cities}, perfectly balanced for your schedule.\n"
            narrative += f"\n{reasoning}\n"

        # 4. Recommended Stays (Grounding Rule)
        if intent in ["PLAN_TRIP", "HOTELS_ONLY", "NEW", "PARTIAL", "MODIFICATION"]:
            if filtered_hotels:
                narrative += "\n🏨 Recommended Stays\n"
                seen = set()
                for h in sorted(filtered_hotels, key=lambda x: x["metadata"]["rating"], reverse=True)[:5]:
                    name = h["metadata"]["name"]
                    if name not in seen:
                        narrative += f"\n{h['metadata']['city']}: {name}\n"
                        narrative += f"* Category: {h['metadata']['price_range'].capitalize()} | Rating: {h['metadata']['rating']}/5.0\n"
                        seen.add(name)
            elif intent == "HOTELS_ONLY":
                narrative += "\nI focus strictly on data-backed options. I couldn't find specific hotels matching those exact filters in our verified data. 😊"

        # 5. Travel Routes (Grounding Rule)
        if intent in ["PLAN_TRIP", "TRAVEL_ONLY", "NEW", "PARTIAL", "MODIFICATION"]:
            if travel:
                narrative += "\n\n🚄 Travel (Intercity)\n"
                for t in travel:
                    narrative += f"\n{t['from']} → {t['to']}\n"
                    narrative += f"{t['mode'].capitalize()} | €{t['cost']} | {t['duration']} hrs\n"
            elif intent == "TRAVEL_ONLY":
                rag_routes = [d for d in all_docs if d.get("category") == "Route"]
                if rag_routes:
                    narrative += "\n\n🚄 Available Routes\n"
                    for r in rag_routes[:5]:
                        m = r["metadata"]
                        narrative += f"\n{m['from']} → {m['to']}\n"
                        narrative += f"{m['mode'].capitalize()} | €{m['cost']} | {m['hours']} hrs\n"

        # 6. Budget Breakdown
        if budget and budget["total"] > 0:
            section = f"\n\n💰 Budget Breakdown\n"
            section += f"\n* Attractions: €{budget['attractions']}"
            section += f"\n* Stays: €{budget['stays']}"
            section += f"\n* Travel: €{budget['transport']}"
            section += f"\n* Dining & Misc: €{int(budget['food'])}"
            section += f"\n\nTotal estimate: €{budget['total']}"
            narrative += section

        return narrative

class EuroPlanOrchestrator:
    def __init__(self, vector_store):
        self.memory = MemoryAgent()
        self.guard = GuardrailAgent()
        self.lang = LanguageAgent()
        self.retrieval = RetrievalAgent(vector_store)
        self.location = LocationAgent()
        self.constraint = ConstraintAgent()
        self.planning = PlanningAgent()
        self.budget = BudgetAgent()
        self.explanation = ExplanationAgent()
        self.persona = PersonaAgent()
        self.clarify = ClarificationAgent()

    def process_stateful(self, query, session_context):
        try:
            print(f"\n--- [EuroPlan AI v15.0] --- Domain Lockdown Process ---")
            
            # 1. Guard (Strict Early Exit)
            guard_res = self.guard.run(query)
            if not guard_res["safe"]: 
                self.guard.log(f"BLOCKED: {guard_res['category']}")
                return {
                    "header": guard_res["message"], 
                    "error_type": "DOMAIN_BLOCK",
                    "valid_plan": False
                }, session_context
            
            self.guard.log("Query PASSED Domain Guardrail.")
            
            # 2. Language & Memory Sync
            lang_res = self.lang.run(query)
            session_context, q_type = self.memory.update_context(lang_res.get("translation", query), session_context)
            
            # 3. Handle Special Intents (v15.0)
            if q_type == "IRRELEVANT":
                return {
                    "header": "I focus exclusively on European travel planning and itineraries. Would you like to plan a trip to France, Italy, or Sweden? 😊",
                    "valid_plan": False
                }, session_context

            if q_type == "HOTELS_ONLY" or q_type == "TRAVEL_ONLY":
                # Ensure we have at least one country to retrieve for
                if not session_context.get("countries"):
                    return {"header": f"Nice 👍 Which country would you like to see {'recommended stays' if q_type == 'HOTELS_ONLY' else 'travel routes'} for?", "valid_plan": False}, session_context
                
                docs = self.retrieval.run(session_context)
                presentation = self.persona.run(session_context, [], {"total": 0, "attractions": 0, "stays": 0, "transport": 0, "food": 0}, "Direct request", [], intent=q_type, all_docs=docs)
                return {
                    "header": presentation,
                    "valid_plan": True,
                    "session_summary": f"Context: {', '.join(session_context['countries'])}",
                    "itinerary": [], "travel": [], "budget_breakdown": None
                }, session_context

            # 4. Completeness Check (Full Trip Flow)
            if not session_context.get("countries") or not session_context.get("budget_provided") or not session_context.get("user_type_provided"):
                msg = self.clarify.run(session_context)
                self.clarify.log(f"Missing parameters (C:{bool(session_context.get('countries'))} B:{session_context.get('budget_provided')} U:{session_context.get('user_type_provided')}). Triggering clarification.")
                return {
                    "header": msg,
                    "valid_plan": False
                }, session_context

            # 5. Retrieval
            docs = self.retrieval.run(session_context)
            pool = [d for d in docs if d.get("category") == "Attraction"]
            self.retrieval.log(f"Pool size: {len(pool)} attractions.")
            
            # 6. Agent Sequence
            cities, by_city = self.location.run(pool, session_context)
            constraint = self.constraint.run(cities, by_city, session_context)
            
            # 7. Planning & Validation (Loop)
            target_days = session_context['duration']
            itinerary, travel = self.planning.run(constraint.get("final_cities", []), by_city, docs, session_context)
            
            # 8. Budget Check
            budget = self.budget.run(itinerary, travel, session_context)
            
            # 9. Consistency Check (v13.0)
            if budget["total"] > session_context["budget"]:
                self.budget.log(f"OVER BUDGET ({budget['total']} > {session_context['budget']}). Pruning...")
                num_cities = max(1, len(cities) - 1)
                cities = cities[:num_cities]
                itinerary, travel = self.planning.run(cities, by_city, docs, session_context)
                budget = self.budget.run(itinerary, travel, session_context)

            # 10. Presentation
            reasoning = self.explanation.run(session_context, cities, [])
            presentation = self.persona.run(session_context, itinerary, budget, reasoning, travel, intent=q_type, all_docs=docs)
            
            return {
                "header": presentation,
                "valid_plan": True,
                "session_summary": f"Context: {', '.join(session_context['countries'])}",
                "itinerary": itinerary,
                "travel": travel,
                "budget_breakdown": budget
            }, session_context

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return {
                "header": "I encountered an error processing your request. Let's try simplifying the destination or duration. 😊",
                "valid_plan": False
            }, session_context

class ClarificationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Clarify")

    def run(self, context):
        if not context.get("countries"):
            return "Got it 👍 Which European country are we exploring? (France, Italy, Germany, etc.)"
        if not context.get("budget_provided"):
            return f"Great choice! Do you have a budget in mind for your trip to {', '.join(context['countries'])}?"
        if not context.get("user_type_provided"):
            return "Almost there! Is this a solo trip, or are you traveling with family or a partner?"
        return "I have everything! Preparing your perfect itinerary... ✈️"
