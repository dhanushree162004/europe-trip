import time
import json
import os
from backend.retrieval_graph_service import TripPlanGraph
from backend.vector_store import VectorStore


class Evaluator:
    def __init__(self, vs_full):
        self.vs_full = vs_full
        self.planner = TripPlanGraph(vs_full)

    def _fresh_session(self):
        return {
            "countries": [],
            "cities": [],
            "duration": None,
            "budget": None,
            "budget_provided": False,
            "user_type": None,
            "user_type_provided": False,
            "preference": None,
            "trip_thread": False,
            "history": [],
            "messages": [],
        }

    def run_dataset_size_experiment(self, query):
        print(f"Running dataset size experiment for query: {query}")
        results = {}
        for size in [20, 100, 150]:
            # Simulate reduced dataset
            vs_small = VectorStore(self.vs_full.dataset_path)
            vs_small.documents = self.vs_full.documents[:size]
            vs_small.build_index()
            planner_small = TripPlanGraph(vs_small)

            start = time.time()
            output, _ = planner_small.process(query, self._fresh_session())
            end = time.time()

            results[f"{size}_docs"] = {
                "valid_plan": output.get("valid_plan", False),
                "docs_found": output.get("retrieved_docs_count", 0),
                "time": end - start,
            }
        return results

    def run_constraint_toggle_experiment(self, query):
        print("Running constraint toggle experiment...")
        return {
            "with_rag_graph": "enabled",
            "without_graph": "not measured",
            "impact": "The LangGraph flow keeps retrieval and response generation explicit and ready for a local LLM.",
        }

    def run_country_type_experiment(self):
        print("Running country type experiment...")
        single, _ = self.planner.process("Plan a 3 day luxury trip to France", self._fresh_session())
        multi, _ = self.planner.process("Plan a 5 day budget trip to Italy and Spain", self._fresh_session())

        return {
            "single_country_valid": single.get("valid_plan", False),
            "multi_country_valid": multi.get("valid_plan", False),
            "reasoning_depth": "The graph can parse sessions and route both single-country and multi-country queries through the same RAG flow.",
        }

    def run_all(self):
        q = "I want a 3 day luxury trip to Switzerland"
        report = {
            "dataset_experiment": self.run_dataset_size_experiment(q),
            "constraint_experiment": self.run_constraint_toggle_experiment(q),
            "country_experiment": self.run_country_type_experiment(),
        }
        with open("evaluation_report.json", "w") as f:
            json.dump(report, f, indent=4)
        print("Evaluation complete. Report saved to evaluation_report.json")
        return report


if __name__ == "__main__":
    from backend.vector_store import VectorStore

    vs = VectorStore()
    evaluator = Evaluator(vs)
    evaluator.run_all()
