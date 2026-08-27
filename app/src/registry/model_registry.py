"""
MongoDB-backed model registry.

Trained models are no longer just ``.pkl`` files sitting in ``app/models``.
Every training run publishes each horizon's model here, where it gets:

* an immutable, monotonically increasing **version** per model name,
* the **metrics** it was evaluated with (MAE / RMSE / R2 + baseline R2),
* the **hyperparameters**, **feature list** and **data lineage** it came from,
* a **lifecycle stage** — ``staging``, ``production`` or ``archived``,
* a checksummed **artifact** in GridFS.

The serving layer asks the registry for whatever is in ``production``, so
promoting or rolling back a model is a metadata change rather than a code
change or a redeploy.

Layout in MongoDB (same database as the feature store):

    model_registry            one document per (name, version)
    model_artifacts.files     GridFS metadata for the pickled models
    model_artifacts.chunks    GridFS binary chunks

Everything here is plain pymongo — no extra service to host and no extra
credentials beyond the ``MONGODB_URI`` the project already uses.
"""

import hashlib
import io
import os
from datetime import datetime, timezone

import joblib
from gridfs import GridFS, NoFile
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from app.src.features.feature_store import get_database

REGISTRY_COLLECTION = "model_registry"

ARTIFACT_BUCKET = "model_artifacts"

STAGE_STAGING = "staging"
STAGE_PRODUCTION = "production"
STAGE_ARCHIVED = "archived"

# Metrics where a smaller number is a better model.
LOWER_IS_BETTER = {"mae", "rmse", "mse"}

# How much worse than the incumbent a candidate may be and still get promoted.
# Each retrain is scored on its own hold-out window, so run-to-run metrics are
# not strictly comparable; this gate is a guard against a badly broken model,
# not a fine-grained comparison.
DEFAULT_DEGRADATION_TOLERANCE = 0.25

# Artifacts kept in GridFS per model name. Metric history is kept forever —
# only the binaries of superseded versions are dropped, so the free-tier
# Atlas cluster does not fill up with a year of daily models.
DEFAULT_ARTIFACTS_KEPT = 5

_indexes_ready = False


def model_name(horizon: str) -> str:
    """Registry name for a forecast horizon, e.g. day1 -> aqi_xgboost_day1."""

    return f"aqi_xgboost_{horizon}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _registry():
    global _indexes_ready

    collection = get_database()[REGISTRY_COLLECTION]

    if not _indexes_ready:
        collection.create_index(
            [("name", ASCENDING), ("version", DESCENDING)],
            unique=True,
        )

        collection.create_index([("name", ASCENDING), ("stage", ASCENDING)])

        _indexes_ready = True

    return collection


def _artifacts() -> GridFS:
    return GridFS(get_database(), collection=ARTIFACT_BUCKET)


def next_version(name: str) -> int:
    latest = _registry().find_one(
        {"name": name},
        sort=[("version", DESCENDING)],
        projection={"version": 1},
    )

    return latest["version"] + 1 if latest else 1


def register_model(
    name: str,
    model,
    *,
    metrics: dict,
    params: dict,
    features: list,
    city: str,
    horizon: str,
    run_id: str,
    data: dict = None,
    environment: dict = None,
    notes: str = None,
    stage: str = STAGE_STAGING,
) -> dict:
    """
    Publish a trained model as the next version of ``name``.

    The artifact is pickled into GridFS with a SHA-256 checksum and the
    metadata document is written afterwards, so a half-finished upload can
    never be picked up as a servable version.
    """

    payload = io.BytesIO()

    joblib.dump(model, payload, compress=3)

    blob = payload.getvalue()

    checksum = hashlib.sha256(blob).hexdigest()

    # The unique (name, version) index is the source of truth for numbering;
    # retry rather than fail if two runs race for the same version.
    for _ in range(3):
        version = next_version(name)

        filename = f"{name}_v{version}.pkl"

        file_id = _artifacts().put(
            blob,
            filename=filename,
            metadata={
                "model_name": name,
                "version": version,
                "run_id": run_id,
                "sha256": checksum,
            },
        )

        document = {
            "name": name,
            "version": version,
            "stage": stage,
            "city": city.lower(),
            "horizon": horizon,
            "run_id": run_id,
            "model_type": type(model).__name__,
            "params": params,
            "features": features,
            "metrics": metrics,
            "data": data or {},
            "environment": environment or {},
            "git_sha": os.getenv("GITHUB_SHA"),
            "artifact": {
                "file_id": file_id,
                "filename": filename,
                "size_bytes": len(blob),
                "sha256": checksum,
                "compressed": True,
            },
            "created_at": _now(),
            "promoted_at": None,
            "notes": notes,
        }

        try:
            _registry().insert_one(dict(document))

        except DuplicateKeyError:
            # Another run claimed this version — drop our orphaned upload
            # and try again with a fresh number.
            _artifacts().delete(file_id)
            continue

        return document

    raise RuntimeError(
        f"Could not allocate a version for '{name}' after 3 attempts."
    )


def get_version(name: str, version: int) -> dict:
    document = _registry().find_one({"name": name, "version": version})

    if document is None:
        raise LookupError(f"'{name}' has no version {version}.")

    return document


def get_production(name: str) -> dict:
    """The production version of ``name``, or None if nothing is promoted."""

    return _registry().find_one({"name": name, "stage": STAGE_PRODUCTION})


def production_versions(names) -> dict:
    """
    ``{name: version}`` for the production stage of several models at once.

    Metadata only — serving uses this to notice a promotion without
    re-downloading the artifacts it already has.
    """

    cursor = _registry().find(
        {"name": {"$in": list(names)}, "stage": STAGE_PRODUCTION},
        projection={"name": 1, "version": 1},
    )

    return {document["name"]: document["version"] for document in cursor}


def list_versions(name: str = None, limit: int = 20) -> list:
    query = {"name": name} if name else {}

    cursor = (
        _registry()
        .find(query)
        .sort([("name", ASCENDING), ("version", DESCENDING)])
        .limit(limit)
    )

    return list(cursor)


def list_names() -> list:
    return sorted(_registry().distinct("name"))


def promote(name: str, version: int) -> dict:
    """Make ``version`` the production model, archiving the incumbent."""

    target = get_version(name, version)

    if target["artifact"].get("file_id") is None:
        raise RuntimeError(
            f"'{name}' v{version} has no artifact left in GridFS "
            f"(pruned on {target['artifact'].get('pruned_at')}); "
            f"it cannot be promoted."
        )

    registry = _registry()

    registry.update_many(
        {
            "name": name,
            "stage": STAGE_PRODUCTION,
            "version": {"$ne": version},
        },
        {"$set": {"stage": STAGE_ARCHIVED}},
    )

    registry.update_one(
        {"name": name, "version": version},
        {"$set": {"stage": STAGE_PRODUCTION, "promoted_at": _now()}},
    )

    return get_version(name, version)


def rollback(name: str) -> dict:
    """
    Re-promote the version that was in production before the current one.

    Only versions that still have their artifact are considered, so a
    rollback can never point serving at a pruned binary.
    """

    current = get_production(name)

    candidates = (
        _registry()
        .find({
            "name": name,
            "promoted_at": {"$ne": None},
            "stage": {"$ne": STAGE_PRODUCTION},
            "artifact.file_id": {"$ne": None},
        })
        .sort("promoted_at", DESCENDING)
        .limit(1)
    )

    previous = list(candidates)

    if not previous:
        raise LookupError(
            f"No previously promoted version of '{name}' is available to "
            f"roll back to."
        )

    if current is not None:
        print(f"Rolling back {name} from v{current['version']}")

    return promote(name, previous[0]["version"])


def load_model(document: dict):
    """Rehydrate the model binary referenced by a registry document."""

    file_id = document["artifact"].get("file_id")

    if file_id is None:
        raise RuntimeError(
            f"{document['name']} v{document['version']} has no artifact "
            f"in GridFS."
        )

    try:
        blob = _artifacts().get(file_id).read()

    except NoFile as exc:
        raise RuntimeError(
            f"Artifact for {document['name']} v{document['version']} is "
            f"missing from GridFS."
        ) from exc

    if hashlib.sha256(blob).hexdigest() != document["artifact"]["sha256"]:
        raise RuntimeError(
            f"Checksum mismatch for {document['name']} "
            f"v{document['version']} — refusing to load."
        )

    return joblib.load(io.BytesIO(blob))


def load_production_model(name: str):
    """Return ``(model, document)`` for whatever is in production."""

    document = get_production(name)

    if document is None:
        raise LookupError(f"'{name}' has no production version.")

    return load_model(document), document


def passes_promotion_gate(
    candidate: dict,
    incumbent: dict = None,
    metric: str = "mae",
    tolerance: float = DEFAULT_DEGRADATION_TOLERANCE,
) -> tuple:
    """
    Decide whether a freshly trained model should take over production.

    Returns ``(promote, reason)``. The first model for a name always wins.
    After that, the candidate is promoted unless it is more than
    ``tolerance`` worse than the incumbent on ``metric`` — fresher data is
    normally worth having, but an obviously broken run should not ship.
    """

    if incumbent is None:
        return True, "no production version yet"

    new_value = candidate.get(metric)
    old_value = incumbent.get("metrics", {}).get(metric)

    if new_value is None or old_value is None:
        return True, f"incumbent has no '{metric}' to compare against"

    if metric in LOWER_IS_BETTER:
        limit = old_value * (1 + tolerance)

        if new_value <= limit:
            return True, (
                f"{metric} {new_value:.4f} within tolerance of "
                f"production {old_value:.4f}"
            )

        return False, (
            f"{metric} {new_value:.4f} is more than {tolerance:.0%} worse "
            f"than production {old_value:.4f}"
        )

    limit = old_value - abs(old_value) * tolerance

    if new_value >= limit:
        return True, (
            f"{metric} {new_value:.4f} within tolerance of "
            f"production {old_value:.4f}"
        )

    return False, (
        f"{metric} {new_value:.4f} is more than {tolerance:.0%} worse "
        f"than production {old_value:.4f}"
    )


def prune_artifacts(name: str, keep: int = DEFAULT_ARTIFACTS_KEPT) -> int:
    """
    Drop the GridFS binaries of superseded versions of ``name``.

    The newest ``keep`` versions and the production version always keep
    theirs. Registry documents are never deleted — the metric history stays
    queryable, the binary is just marked as pruned.
    """

    registry = _registry()

    versions = list(
        registry
        .find(
            {"name": name, "artifact.file_id": {"$ne": None}},
            projection={"version": 1, "stage": 1, "artifact.file_id": 1},
        )
        .sort("version", DESCENDING)
    )

    pruned = 0

    for document in versions[keep:]:
        if document["stage"] == STAGE_PRODUCTION:
            continue

        _artifacts().delete(document["artifact"]["file_id"])

        registry.update_one(
            {"_id": document["_id"]},
            {
                "$set": {
                    "artifact.file_id": None,
                    "artifact.pruned_at": _now(),
                }
            },
        )

        pruned += 1

    return pruned


def summarise(document: dict) -> dict:
    """A JSON-safe view of a registry document, without the raw ids."""

    if document is None:
        return None

    artifact = document.get("artifact", {})

    return {
        "name": document["name"],
        "version": document["version"],
        "stage": document["stage"],
        "city": document.get("city"),
        "horizon": document.get("horizon"),
        "run_id": document.get("run_id"),
        "model_type": document.get("model_type"),
        "metrics": document.get("metrics", {}),
        "params": document.get("params", {}),
        "feature_count": len(document.get("features", [])),
        "data": document.get("data", {}),
        "environment": document.get("environment", {}),
        "git_sha": document.get("git_sha"),
        "artifact": {
            "filename": artifact.get("filename"),
            "size_bytes": artifact.get("size_bytes"),
            "sha256": artifact.get("sha256"),
            "available": artifact.get("file_id") is not None,
        },
        "created_at": document.get("created_at"),
        "promoted_at": document.get("promoted_at"),
        "notes": document.get("notes"),
    }
