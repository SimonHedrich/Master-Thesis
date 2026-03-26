# Remote server configuration
# REMOTE_HOST := gpu-server.taile550ef.ts.net
REMOTE_HOST := gpu.local
REMOTE_PATH := ~/Master-Thesis/

# Sync large files excluded by .gitignore to the remote server.
# Uses --ignore-existing to skip files already present on the remote.
sync:
	rsync -avh --progress --ignore-existing \
		data/ \
		$(REMOTE_HOST):$(REMOTE_PATH)data/
	rsync -avh --progress --ignore-existing \
		resources/SNPredictions_all* \
		$(REMOTE_HOST):$(REMOTE_PATH)resources/
