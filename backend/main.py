from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.vector_store import VectorStore
from backend.retrieval_graph_service import TripPlanGraph
from backend.evaluation import Evaluator
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="EuroPlan AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
DATASET_PATH = os.getenv("DATASET_PATH")
vs = VectorStore(DATASET_PATH)
planner = TripPlanGraph(vs)
evaluator = Evaluator(vs)

# Global session store
sessions = {}


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"


@app.get("/")
def read_root():
    return {"status": "online", "message": "LangGraph Travel Planner API"}


@app.post("/plan")
def generate_plan(request: QueryRequest):
    try:
        # Initialize session if not exists
        if request.session_id not in sessions:
            sessions[request.session_id] = {
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

        session = sessions[request.session_id]

        # Process through the LangGraph pipeline
        result, updated_context = planner.process(request.query, session)

        # Save updated context
        sessions[request.session_id] = updated_context

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluate")
def run_evaluation():
    try:
        report = evaluator.run_all()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
