# Remote server configuration
# REMOTE_HOST := gpu-server.taile550ef.ts.net
REMOTE_HOST := gpu.local
REMOTE_PATH := ~/Master-Thesis/

# Set to true to skip files already present on the remote, false to overwrite
IGNORE_EXISTING := true

# Resolve rsync flag
ifeq ($(IGNORE_EXISTING),true)
  _IGNORE_FLAG := --ignore-existing
else
  _IGNORE_FLAG :=
endif

_RSYNC := rsync -avh --progress $(_IGNORE_FLAG)

# Sync large files excluded by .gitignore to the remote server.
# Add new paths here as additional rsync lines.
sync:
# 	$(_RSYNC) data/ $(REMOTE_HOST):$(REMOTE_PATH)data/
# 	$(_RSYNC) resources/SNPredictions_all* $(REMOTE_HOST):$(REMOTE_PATH)resources/
	$(_RSYNC) reports/wikimedia_categories/* $(REMOTE_HOST):$(REMOTE_PATH)reports/wikimedia_categories/

# Sync all gitignored files to the remote server.
# Uses git to discover ignored files, then rsyncs each one preserving directory structure.
sync-ignored:
	@git ls-files --others --ignored --exclude-standard | rsync -avh --progress --ignore-existing --files-from=- . $(REMOTE_HOST):$(REMOTE_PATH)
