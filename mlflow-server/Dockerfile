# Pinned to the exact version the Hetzner server ran (image built 2026-04-06 from
# :latest). Bumping this will run Alembic schema migrations on mlflow.db -- back it
# up first.
FROM ghcr.io/mlflow/mlflow:v3.10.1

RUN pip install mlflow[auth]

# On the server the backend store was never set, so MLflow silently used
# sqlite:///mlflow.db in the container's writable layer (not a mount). Here it is
# explicit and bind-mounted so the data lives in this directory.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/db/mlflow.db", \
     "--default-artifact-root", "/mlartifacts", \
     "--app-name", "basic-auth", "--allowed-hosts", "*"]
