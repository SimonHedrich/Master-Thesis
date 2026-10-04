# mlflow-server — local MLflow tracking server

The thesis's MLflow tracking server, migrated on 2026-10-05 from the Hetzner VPS
(`hetzner.taile550ef.ts.net:5000`, `~/personal_server/apps/mlflow/`) to this machine
so the VPS can be decommissioned. All experiments, runs, metrics, params, artifacts
and the basic-auth user database were copied verbatim (checksums verified).

## Run

```
cd mlflow-server
docker compose up -d --build      # UI: http://localhost:5050  (basic auth, user: simon)
docker compose logs -f            # follow
docker compose stop               # stop; `down` also removes the container (data is on disk, so safe)
```

Host port is **5050**, not 5000: macOS AirPlay Receiver occupies 5000 and answers
every request with an empty 403 (`Server: AirTunes`). Training `.env` files that
still point at `http://hetzner.taile550ef.ts.net:5000` must be changed to the new
host, e.g. `http://localhost:5050` on this machine, or
`http://macbook-pro.taile550ef.ts.net:5050` from another tailnet node.

## Layout (everything except the four config files is gitignored)

| path                      | what                                               | tracked |
|---------------------------|----------------------------------------------------|---------|
| `compose.yaml`            | service definition                                 | yes     |
| `Dockerfile`              | `ghcr.io/mlflow/mlflow:v3.10.1` + `mlflow[auth]`    | yes     |
| `.env.example`            | template for `.env` (Flask session secret)         | yes     |
| `.env`                    | the actual secret, copied from the server          | no      |
| `config/auth_config.ini`  | basic-auth settings incl. admin password           | no      |
| `config/auth.db`          | basic-auth user/permission SQLite DB               | no      |
| `db/mlflow.db`            | tracking backend store (SQLite, ~180 MB)           | no      |
| `db/mlflow.db.bak-*`      | pre-first-start backup of the above                | no      |
| `mlartifacts/`            | artifact store (~8.8 GB, served via the server)    | no      |
| `mlruns/`                 | empty legacy mount, kept for parity                | no      |

## Differences from the Hetzner setup

- **Backend store is now explicit.** On the server `mlflow server` was started
  without `--backend-store-uri`, so MLflow 3 defaulted to `sqlite:///mlflow.db` in
  the container's *writable layer* — not in any bind mount. The 183 MB database was
  only recoverable with `docker cp mlflow:/mlflow.db` from the stopped container; a
  `docker compose down` would have destroyed it. Here it is `sqlite:////mlflow/db/mlflow.db`
  on a bind-mounted directory.
- **Image pinned to `v3.10.1`**, the version the server's `:latest` resolved to when
  it was built (2026-04-06). Bumping the tag will run Alembic migrations on
  `db/mlflow.db` on first start; copy the file first. The pin was chosen so the
  migration left the database byte-identical (verified by checksum).
- **Host port 5050** and `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS=http://localhost:5050`
  (see above).
- Artifact URIs in the DB are `mlflow-artifacts:/<exp>/<run>/artifacts`, i.e. proxied
  through the server, so no path rewriting was needed.

## State at migration (2026-10-05)

5 experiments (`yolov5s`, `yolo26n-wildlife225`, `yolo26n-synthetic-model-comparison`,
`teacher-finetune-speciesnet225`, `Default`), 82 runs (81 active, 1 deleted),
847 138 metric rows, 5 145 params, 7 587 artifact files. Alembic revision
`1b5f0d9ad7c1`. Model registry unused.
