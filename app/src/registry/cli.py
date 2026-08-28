"""
Command line access to the model registry.

    python -m app.src.registry.cli list
    python -m app.src.registry.cli list --name aqi_xgboost_day3 --limit 10
    python -m app.src.registry.cli show aqi_xgboost_day1
    python -m app.src.registry.cli show aqi_xgboost_day1 --version 2
    python -m app.src.registry.cli promote aqi_xgboost_day3 4
    python -m app.src.registry.cli rollback aqi_xgboost_day3
    python -m app.src.registry.cli prune --keep 5
    python -m app.src.registry.cli download aqi_xgboost_day1 --out day1.pkl
"""

import argparse
import json

import joblib

from app.src.registry import model_registry as registry

HEADER = (
    f"{'NAME':<22}{'VER':>4}  {'STAGE':<11}{'MODEL':<15}"
    f"{'MAE':>8}{'RMSE':>8}{'R2':>9}{'BASE R2':>9}"
    f"{'ROWS':>8}  {'TRAINED AT':<22}ARTIFACT"
)


def _row(document: dict) -> str:
    metrics = document.get("metrics", {})

    def number(value, width, digits=2):
        if value is None:
            return f"{'-':>{width}}"

        return f"{value:>{width}.{digits}f}"

    # Versions registered before candidate selection have no `candidate`
    # field; fall back to the estimator class so old rows still line up.
    model = document.get("candidate") or document.get("model_type") or "-"

    return (
        f"{document['name']:<22}"
        f"{document['version']:>4}  "
        f"{document['stage']:<11}"
        f"{model[:14]:<15}"
        f"{number(metrics.get('mae'), 8)}"
        f"{number(metrics.get('rmse'), 8)}"
        f"{number(metrics.get('r2'), 9, 4)}"
        f"{number(metrics.get('baseline_r2'), 9, 4)}"
        f"{document.get('data', {}).get('usable_rows', 0):>8}  "
        f"{str(document.get('created_at', '-'))[:19]:<22}"
        f"{'ok' if document['artifact'].get('file_id') else 'pruned'}"
    )


def cmd_list(args) -> None:
    documents = registry.list_versions(name=args.name, limit=args.limit)

    if not documents:
        print("Registry is empty.")
        return

    print(HEADER)

    for document in documents:
        print(_row(document))


def cmd_show(args) -> None:
    if args.version is None:
        document = registry.get_production(args.name)

        if document is None:
            raise SystemExit(f"'{args.name}' has no production version.")

    else:
        document = registry.get_version(args.name, args.version)

    print(json.dumps(registry.summarise(document, full=True), indent=2))


def cmd_promote(args) -> None:
    document = registry.promote(args.name, args.version)

    print(
        f"{document['name']} v{document['version']} is now "
        f"{document['stage']} (promoted at {document['promoted_at']})."
    )


def cmd_rollback(args) -> None:
    document = registry.rollback(args.name)

    print(
        f"Rolled back to {document['name']} v{document['version']} "
        f"(trained {document['created_at']})."
    )


def cmd_prune(args) -> None:
    names = [args.name] if args.name else registry.list_names()

    for name in names:
        removed = registry.prune_artifacts(name, keep=args.keep)

        print(f"{name}: pruned {removed} artifact(s), kept newest {args.keep}.")


def cmd_download(args) -> None:
    if args.version is None:
        document = registry.get_production(args.name)

        if document is None:
            raise SystemExit(f"'{args.name}' has no production version.")

    else:
        document = registry.get_version(args.name, args.version)

    model = registry.load_model(document)

    out = args.out or document["artifact"]["filename"]

    joblib.dump(model, out)

    print(f"Wrote {document['name']} v{document['version']} to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and manage the AQI model registry",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list", help="List registered versions")
    listing.add_argument("--name", help="Only this model name")
    listing.add_argument("--limit", type=int, default=20)
    listing.set_defaults(func=cmd_list)

    show = subparsers.add_parser("show", help="Show one version in full")
    show.add_argument("name")
    show.add_argument(
        "--version",
        type=int,
        help="Defaults to the production version",
    )
    show.set_defaults(func=cmd_show)

    promote = subparsers.add_parser("promote", help="Promote a version")
    promote.add_argument("name")
    promote.add_argument("version", type=int)
    promote.set_defaults(func=cmd_promote)

    rollback = subparsers.add_parser(
        "rollback",
        help="Re-promote the previous production version",
    )
    rollback.add_argument("name")
    rollback.set_defaults(func=cmd_rollback)

    prune = subparsers.add_parser(
        "prune",
        help="Drop artifacts of superseded versions (metrics are kept)",
    )
    prune.add_argument("--name", help="Only this model name")
    prune.add_argument(
        "--keep",
        type=int,
        default=registry.DEFAULT_ARTIFACTS_KEPT,
    )
    prune.set_defaults(func=cmd_prune)

    download = subparsers.add_parser(
        "download",
        help="Write a registered model back out as a local .pkl",
    )
    download.add_argument("name")
    download.add_argument("--version", type=int)
    download.add_argument("--out")
    download.set_defaults(func=cmd_download)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
