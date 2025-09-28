#!/usr/bin/env python3
"""
Test runner for QuantumHybridDFT project.
Provides a convenient way to run all tests with enhanced output and options.
"""

import argparse
import sys
import time
import unittest
from pathlib import Path


def print_separator(char="-", length=70):
    """Print a separator line."""
    print(char * length)


def run_tests(test_dir="tests", pattern="test_*.py", verbosity=2, failfast=False):
    """
    Run all tests in the specified directory.

    Args:
        test_dir: Directory containing test files
        pattern: Pattern to match test files
        verbosity: Verbosity level (0=quiet, 1=normal, 2=verbose)
        failfast: Stop on first failure

    Returns:
        TestResult object
    """
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)

    # Count total tests
    total_tests = suite.countTestCases()

    # Create test runner
    runner = unittest.TextTestRunner(
        verbosity=verbosity, failfast=failfast, stream=sys.stdout
    )

    # Print header
    print_separator("=")
    print(f"Running {total_tests} tests from {test_dir}/")
    print_separator("=")
    print()

    # Run tests and measure time
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()

    # Print summary
    print()
    print_separator("=")
    print("TEST SUMMARY")
    print_separator("-")
    print(f"Total tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print_separator("=")

    # Print failed tests details if any
    if result.failures or result.errors:
        print()
        print("FAILED TESTS:")
        print_separator("-")

        for test, traceback in result.failures + result.errors:
            print(f"\n{test}:")
            print(traceback)

    return result


def list_test_files(test_dir="tests", pattern="test_*.py"):
    """List all test files found."""
    test_path = Path(test_dir)
    test_files = sorted(test_path.glob(pattern))

    print_separator("=")
    print("Available test files:")
    print_separator("-")

    for test_file in test_files:
        # Try to get test count for each file
        loader = unittest.TestLoader()
        try:
            suite = loader.discover(test_dir, pattern=test_file.name)
            count = suite.countTestCases()
            print(f"  - {test_file.name:<30} ({count} tests)")
        except:
            print(f"  - {test_file.name}")

    print_separator("=")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run tests for QuantumHybridDFT project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                  # Run all tests
  python run_tests.py -v               # Run with verbose output
  python run_tests.py -q               # Run with quiet output
  python run_tests.py -f               # Stop on first failure
  python run_tests.py --list           # List all test files
  python run_tests.py -p test_scf.py   # Run only test_scf.py
        """,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output (show each test)"
    )

    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Quiet output (minimal information)"
    )

    parser.add_argument(
        "-f", "--failfast", action="store_true", help="Stop on first failure"
    )

    parser.add_argument(
        "--list", action="store_true", help="List all test files and exit"
    )

    parser.add_argument(
        "-p",
        "--pattern",
        default="test_*.py",
        help="Pattern to match test files (default: test_*.py)",
    )

    parser.add_argument(
        "-d",
        "--directory",
        default="tests",
        help="Directory containing tests (default: tests)",
    )

    args = parser.parse_args()

    # Handle list option
    if args.list:
        list_test_files(args.directory, args.pattern)
        return 0

    # Determine verbosity
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1

    # Run tests
    result = run_tests(
        test_dir=args.directory,
        pattern=args.pattern,
        verbosity=verbosity,
        failfast=args.failfast,
    )

    # Return appropriate exit code
    if result.wasSuccessful():
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
