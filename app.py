from fastapi import FastAPI

from environments.email_triage_env import EmailTriageEnv
from openenv.models import Action
from openenv.tasks import get_benchmark_tasks

app = FastAPI()
task = get_benchmark_tasks()[0]
env = EmailTriageEnv(task=task)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": str(obs)}


@app.post("/step")
def step(action: Action):
    observation, reward, done, info = env.step(action)
    return {
        "observation": str(observation),
        "reward": reward.total,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state()
