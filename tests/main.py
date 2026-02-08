#!/usr/bin/env python3
"""
Test Runner for torch_amt Test Suite

This script runs all tests in the analysis and/or functional directories using pytest.
Provides verbose output, progress monitoring, and saves logs with timestamps.

Usage:
    # Run all tests
    python main.py
    
    # Run only functional tests
    python main.py --type functional
    
    # Run only analysis tests
    python main.py --type analysis
    
    # Run specific test file
    python main.py --file tests/analysis/test_headphone_filter.py
    
    # Run with custom pytest args
    python main.py --type functional --pytest-args "-k cpu -x"

Features:
    - Real-time verbose output with progress monitoring
    - Automatic log file creation (logs/test_run_YYYYMMDD_HHMMSS.log)
    - Captures warnings, errors, and full pytest output
    - Summary report at the end
    - Exit code reflects test success/failure
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional


# ================================================================================================
# Configuration
# ================================================================================================

TESTS_DIR = Path(__file__).parent
ANALYSIS_DIR = TESTS_DIR / "analysis"
FUNCTIONAL_DIR = TESTS_DIR / "functional"
LOGS_DIR = TESTS_DIR.parent / "logs"

# Create logs directory
LOGS_DIR.mkdir(exist_ok=True)


# ================================================================================================
# Test Discovery
# ================================================================================================

def find_test_files(test_type: str = "all") -> List[Path]:
    """
    Find all test files based on test type.
    
    Parameters
    ----------
    test_type : str
        Type of tests to find: 'analysis', 'functional', or 'all'
        
    Returns
    -------
    list of Path
        List of test file paths
    """
    test_files = []
    
    if test_type in ["analysis", "all"]:
        analysis_tests = sorted(ANALYSIS_DIR.glob("test_*.py"))
        test_files.extend(analysis_tests)
    
    if test_type in ["functional", "all"]:
        functional_tests = sorted(FUNCTIONAL_DIR.glob("test_*.py"))
        test_files.extend(functional_tests)
    
    return test_files


# ================================================================================================
# Pytest Execution
# ================================================================================================

def run_pytest(test_paths: List[Path], pytest_args: Optional[List[str]] = None, 
               log_file: Optional[Path] = None) -> int:
    """
    Run pytest on specified test files with real-time output.
    
    Parameters
    ----------
    test_paths : list of Path
        List of test file paths to run
    pytest_args : list of str, optional
        Additional pytest arguments
    log_file : Path, optional
        Path to log file for saving output
        
    Returns
    -------
    int
        Exit code (0 = success, non-zero = failure)
    """
    if not test_paths:
        print("✗ No test files found!")
        return 1
    
    # Build pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",  # Verbose
        "--tb=short",  # Short traceback format
        "--color=yes",  # Colored output
    ]
    
    # Add custom pytest args if provided
    if pytest_args:
        cmd.extend(pytest_args)
    
    # Add test paths
    cmd.extend([str(p) for p in test_paths])
    
    print(f"\n{'='*80}")
    print(f"RUNNING PYTEST")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Test files: {len(test_paths)}")
    for p in test_paths:
        print(f"  - {p.relative_to(TESTS_DIR.parent)}")
    if log_file:
        print(f"Log file: {log_file.relative_to(TESTS_DIR.parent)}")
    print(f"{'='*80}\n")
    
    # Run pytest with real-time output
    try:
        if log_file:
            # Tee output to both console and log file
            with open(log_file, 'w') as f:
                # Write header to log
                f.write(f"Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"{'='*80}\n\n")
                f.flush()
                
                # Run pytest with stdout/stderr redirected
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Stream output to console and file
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
                    f.flush()
                
                # Wait for completion
                returncode = process.wait()
        else:
            # Just run pytest with direct output (no log file)
            result = subprocess.run(cmd)
            returncode = result.returncode
        
        return returncode
    
    except KeyboardInterrupt:
        print("\n\n✗ Tests interrupted by user (Ctrl+C)")
        return 130
    except Exception as e:
        print(f"\n\n✗ Error running pytest: {e}")
        return 1


# ================================================================================================
# Main Execution
# ================================================================================================

def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Run torch_amt test suite with pytest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run all tests
  python main.py --type functional                  # Run only functional tests
  python main.py --type analysis                    # Run only analysis tests
  python main.py --file tests/analysis/test_*.py    # Run specific test file
  python main.py --type functional --pytest-args "-k cpu -x"  # Custom pytest args
        """
    )
    
    parser.add_argument(
        "--type",
        choices=["analysis", "functional", "all"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Specific test file to run (overrides --type)"
    )
    
    parser.add_argument(
        "--pytest-args",
        type=str,
        help="Additional arguments to pass to pytest (space-separated, quoted)"
    )
    
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable log file creation"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("TORCH_AMT TEST SUITE RUNNER")
    print("="*80)
    
    # Find test files
    if args.file:
        # Run specific file
        test_file = Path(args.file)
        if not test_file.exists():
            print(f"✗ Test file not found: {test_file}")
            return 1
        test_paths = [test_file]
        print(f"Mode: Single file")
        print(f"File: {test_file}")
    else:
        # Run tests based on type
        test_paths = find_test_files(args.type)
        print(f"Mode: {args.type.upper()} tests")
        print(f"Found: {len(test_paths)} test files")
    
    if not test_paths:
        print("✗ No test files found!")
        return 1
    
    # Parse additional pytest args
    pytest_args = args.pytest_args.split() if args.pytest_args else []
    
    # Create log file with timestamp
    if args.no_log:
        log_file = None
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"test_run_{timestamp}.log"
    
    # Run pytest
    start_time = datetime.now()
    returncode = run_pytest(test_paths, pytest_args, log_file)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST RUN SUMMARY")
    print(f"{'='*80}")
    print(f"Duration: {duration:.1f} seconds")
    if log_file:
        print(f"Log file: {log_file.relative_to(TESTS_DIR.parent)}")
    
    if returncode == 0:
        print("Status: ✓ ALL TESTS PASSED")
    else:
        print(f"Status: ✗ TESTS FAILED (exit code: {returncode})")
    print(f"{'='*80}\n")
    
    return returncode


if __name__ == "__main__":
    sys.exit(main())
