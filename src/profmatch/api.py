"""Optional HTTP API (extra `api`): GET / (a small form), GET /health, POST /recommend.

No debug mode. CORS is off unless PROFMATCH_CORS_ORIGINS lists exact origins.
This module has no `from __future__ import annotations`: FastAPI must see the real `Body` class.
"""

from .config import Settings
from .recommend import Query, QueryError, Recommender
from .store import load
from .textproc import CourseCodeError

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>profmatch</title>
<style>body{font-family:system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem}
input,select,button{font-size:1rem;margin:.25rem 0;padding:.4rem;width:100%}li{margin:.6rem 0}</style></head>
<body><h1>profmatch</h1><form id="f"><label>Interests<input name="interests" required></label>
<label>Course code (optional)<input name="course" placeholder="ABCD 1234"></label>
<label>Challenge<select name="challenge"><option value="">no preference</option><option>lower</option>
<option>balanced</option><option>higher</option></select></label><button>Recommend</button></form>
<p id="notes"></p><ol id="out"></ol>
<script>
document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const d = Object.fromEntries(new FormData(e.target));
  const r = await fetch('recommend', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({interests: d.interests, course: d.course || null, challenge: d.challenge || null})});
  const j = await r.json(); const out = document.getElementById('out'); out.textContent = '';
  document.getElementById('notes').textContent = r.ok ? j.notes.join(' ') : (j.detail || 'error');
  for (const x of (j.recommendations || [])) {
    const li = document.createElement('li');
    li.textContent = `${x.name} (${x.title}), score ${x.score}: ${x.reasons.join('; ')}`;
    out.appendChild(li);
  }
};
</script></body></html>"""


def create_app(rec: Recommender | None = None, settings: Settings | None = None):
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field

    settings = settings or Settings.from_env()
    if rec is None:
        rec = Recommender(load(settings.db_path), settings.matcher, settings.weights, settings.prior_strength,
                          settings.min_responses, settings.display)

    class Body(BaseModel):
        interests: str = Field("", max_length=500)
        course: str | None = Field(None, max_length=20)
        challenge: str | None = None
        k: int = Field(5, ge=1, le=20)

    app = FastAPI(title="profmatch", debug=False)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type"])

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE

    @app.get("/health")
    def health():
        return {"status": "ok", "professors": len(rec.cat.professors)}

    @app.post("/recommend")
    def recommend(body: Body):
        try:
            return rec.recommend(Query(body.interests, body.course, body.challenge, body.k)).to_dict()
        except (QueryError, CourseCodeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
