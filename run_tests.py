#!/usr/bin/env python
"""
CHAL Test Suite Runner

Cross-platform test runner for the CHAL framework.
Runs unit, integration, and end-to-end tests with coverage reporting.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --unit       # Run unit tests only
    python run_tests.py --integration # Run integration tests only
    python run_tests.py --e2e        # Run end-to-end tests only
    python run_tests.py --coverage   # Generate HTML coverage report
"""

import sys
import subprocess
from pathlib import Path
import argparse


def run_command(cmd, description, fail_fast=True):
    """Run a command and handle errors.

    Args:
        cmd: Command list to execute
        description: Human-readable description
        fail_fast: If True, exit on failure; if False, continue

    Returns:
        bool: True if command succeeded, False otherwise
    """
    print(f"\n{'=' * 70}")
    print(f"[{description}]")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print(f"\n❌ {description} failed!")
        if fail_fast:
            sys.exit(1)
        return False

    print(f"\n✅ {description} passed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="CHAL Test Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --unit             # Run unit tests only
  python run_tests.py --integration      # Run integration tests only
  python run_tests.py --e2e              # Run end-to-end tests only
  python run_tests.py --coverage         # Generate HTML coverage report
  python run_tests.py --verbose          # Verbose output
        """
    )

    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only"
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Run end-to-end tests only"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose test output"
    )
    parser.add_argument(
        "--no-cov",
        action="store_true",
        help="Disable coverage reporting"
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" " * 20 + "CHAL Test Suite Runner")
    print("=" * 70)

    # Determine which tests to run
    run_unit = args.unit or not (args.integration or args.e2e)
    run_integration = args.integration or not (args.unit or args.e2e)
    run_e2e = args.e2e or not (args.unit or args.integration)

    # Build base pytest command
    base_cmd = ["pytest"]
    if args.verbose:
        base_cmd.append("-v")

    # Add coverage if not disabled
    cov_args = []
    if not args.no_cov:
        cov_args = ["--cov=src/chal", "--cov-report=term-missing"]
        if args.coverage:
            cov_args.append("--cov-report=html")

    success = True

    # Run unit tests
    if run_unit:
        cmd = base_cmd + ["tests/", "-m", "unit"] + cov_args
        if not run_command(cmd, "Unit Tests"):
            success = False

    # Run integration tests
    if run_integration:
        cmd = base_cmd + ["tests/integration/", "-m", "integration"]
        if not run_command(cmd, "Integration Tests", fail_fast=False):
            success = False

    # Run end-to-end tests
    if run_e2e:
        cmd = base_cmd + ["tests/e2e/", "-m", "e2e"]
        if not run_command(cmd, "End-to-End Tests", fail_fast=False):
            success = False

    # Final summary
    print("\n" + "=" * 70)
    if success:
        print("✅ All Tests Passed!")
    else:
        print("❌ Some Tests Failed")
    print("=" * 70)

    if args.coverage and not args.no_cov:
        print("\n📊 Coverage report generated: htmlcov/index.html")
        print("   View with: open htmlcov/index.html (Mac)")
        print("              xdg-open htmlcov/index.html (Linux)")
        print("              start htmlcov/index.html (Windows)")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
