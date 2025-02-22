from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ctransformers import AutoModelForCausalLM

MODEL_PATH = "C:\\Users\\ankit\\Downloads\\llama-2-7b-chat.Q4_K_M.gguf"

app = FastAPI()

# ✅ Fix CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Try allowing all origins for testing
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # ✅ Explicitly allow OPTIONS
    allow_headers=["*"],  # ✅ Allow all headers
)

# Load the model
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    model_type="llama",
    max_new_tokens=256,
    temperature=0.7
)

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: QueryRequest):
    response = llm(request.query)
    return {"response": response}

# Run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
