import argparse
import json
import os

from simulator.vehicle_model.validators import TeacherEpisodeValidator, write_validation_report


def validate_dataset(dataset_dir: str, report_path: str) -> int:
    validator = TeacherEpisodeValidator()
    report = validator.validate_dataset(dataset_dir)
    write_validation_report(report, report_path)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--schema", required=False)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    report_path = args.report or os.path.join(args.dataset, "validation_report.json")
    return validate_dataset(args.dataset, report_path)


if __name__ == "__main__":
    raise SystemExit(main())
