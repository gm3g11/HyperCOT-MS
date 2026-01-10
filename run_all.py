#!/usr/bin/env python3
"""
HyperCOT-MS Pipeline Script

This script runs the complete HyperCOT pipeline for comparing Morse-Smale complexes:
1. Generate hypergraph structure from MS complex data
2. Compute node measures (μ) using extended persistence
3. Compute hyperedge measures (ν)
4. Compute hypernetwork distance function (ω)
5. Run HyperCOT optimization
6. Generate visualizations

Usage:
    python run_all.py                    # Run full pipeline with defaults
    python run_all.py --config my.yaml   # Use custom config
    python run_all.py --steps 1,2,3      # Run specific steps only
    python run_all.py --clean-only       # Only compute for clean data
    python run_all.py -v                 # Verbose output

Author: HyperCOT-MS Team
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    get_project_root,
    load_config,
    setup_logging,
    get_logger,
    resolve_path,
    HyperCOTError
)


# =============================================================================
# Pipeline Steps
# =============================================================================

def step1_generate_hypergraph(config: dict, logger) -> bool:
    """Step 1: Generate hypergraph structure from MS complex data."""
    logger.info("=" * 60)
    logger.info("STEP 1: Generating Hypergraph Structure")
    logger.info("=" * 60)

    try:
        import generate_hypergraph
        generate_hypergraph.main()
        logger.info("✓ Hypergraph generation completed")
        return True
    except Exception as e:
        logger.error(f"✗ Hypergraph generation failed: {e}")
        return False


def step2_compute_node_measure(config: dict, logger) -> bool:
    """Step 2: Compute node measures (μ) using extended persistence."""
    logger.info("=" * 60)
    logger.info("STEP 2: Computing Node Measures (μ)")
    logger.info("=" * 60)

    try:
        import compute_node_measure
        compute_node_measure.main()
        logger.info("✓ Node measure computation completed")
        return True
    except Exception as e:
        logger.error(f"✗ Node measure computation failed: {e}")
        return False


def step3_compute_nu(config: dict, logger) -> bool:
    """Step 3: Compute hyperedge measures (ν)."""
    logger.info("=" * 60)
    logger.info("STEP 3: Computing Hyperedge Measures (ν)")
    logger.info("=" * 60)

    try:
        import compute_nu
        compute_nu.main()
        logger.info("✓ Hyperedge measure computation completed")
        return True
    except Exception as e:
        logger.error(f"✗ Hyperedge measure computation failed: {e}")
        return False


def step4_compute_omega(config: dict, logger) -> bool:
    """Step 4: Compute hypernetwork distance function (ω)."""
    logger.info("=" * 60)
    logger.info("STEP 4: Computing Hypernetwork Function (ω)")
    logger.info("=" * 60)

    try:
        import compute_omega
        compute_omega.main()
        logger.info("✓ Hypernetwork function computation completed")
        return True
    except Exception as e:
        logger.error(f"✗ Hypernetwork function computation failed: {e}")
        return False


def step5_run_hypercot(config: dict, logger) -> bool:
    """Step 5: Run HyperCOT optimization."""
    logger.info("=" * 60)
    logger.info("STEP 5: Running HyperCOT Optimization")
    logger.info("=" * 60)

    try:
        import compute_hypercot
        compute_hypercot.main()
        logger.info("✓ HyperCOT optimization completed")
        return True
    except Exception as e:
        logger.error(f"✗ HyperCOT optimization failed: {e}")
        return False


def step6_visualize(config: dict, logger) -> bool:
    """Step 6: Generate visualizations."""
    logger.info("=" * 60)
    logger.info("STEP 6: Generating Visualizations")
    logger.info("=" * 60)

    try:
        import visualize_hypercot_final
        visualize_hypercot_final.main()
        logger.info("✓ Visualization generation completed")
        return True
    except Exception as e:
        logger.error(f"✗ Visualization generation failed: {e}")
        return False


# Pipeline step registry
PIPELINE_STEPS = {
    1: ("Generate Hypergraph", step1_generate_hypergraph),
    2: ("Compute Node Measure (μ)", step2_compute_node_measure),
    3: ("Compute Hyperedge Measure (ν)", step3_compute_nu),
    4: ("Compute Hypernetwork Function (ω)", step4_compute_omega),
    5: ("Run HyperCOT Optimization", step5_run_hypercot),
    6: ("Generate Visualizations", step6_visualize),
}


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(
    steps: Optional[List[int]] = None,
    config_path: Optional[str] = None,
    verbose: bool = False
) -> bool:
    """
    Run the HyperCOT pipeline.

    Args:
        steps: List of step numbers to run. None runs all steps.
        config_path: Path to configuration file.
        verbose: Enable verbose output.

    Returns:
        True if all steps succeeded, False otherwise.
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logger = setup_logging(level=log_level, log_file="output/hypercot.log")

    logger.info("=" * 60)
    logger.info("HyperCOT-MS Pipeline")
    logger.info("Morse-Smale Complex Comparison using Optimal Transport")
    logger.info("=" * 60)

    # Load configuration
    try:
        config = load_config(config_path)
        logger.info(f"Configuration loaded from: {config_path or 'config.yaml'}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return False

    # Determine which steps to run
    if steps is None:
        steps = list(PIPELINE_STEPS.keys())
    else:
        invalid = [s for s in steps if s not in PIPELINE_STEPS]
        if invalid:
            logger.error(f"Invalid step numbers: {invalid}")
            logger.info(f"Valid steps: {list(PIPELINE_STEPS.keys())}")
            return False

    logger.info(f"Running steps: {steps}")
    logger.info("")

    # Run pipeline
    start_time = time.time()
    results = {}

    for step_num in steps:
        step_name, step_func = PIPELINE_STEPS[step_num]
        step_start = time.time()

        logger.info(f"Starting Step {step_num}: {step_name}")
        success = step_func(config, logger)
        results[step_num] = success

        step_time = time.time() - step_start
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{status} - Step {step_num} completed in {step_time:.1f}s")
        logger.info("")

        if not success:
            logger.warning(f"Step {step_num} failed, but continuing with remaining steps...")

    # Summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)

    for step_num in steps:
        step_name, _ = PIPELINE_STEPS[step_num]
        status = "✓" if results.get(step_num, False) else "✗"
        logger.info(f"  Step {step_num}: {status} {step_name}")

    succeeded = sum(results.values())
    total = len(steps)
    logger.info("")
    logger.info(f"Completed: {succeeded}/{total} steps")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info("=" * 60)

    return all(results.values())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HyperCOT-MS Pipeline: Compare Morse-Smale complexes using optimal transport",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                    # Run full pipeline
  python run_all.py --steps 5,6        # Only run HyperCOT and visualization
  python run_all.py -v                 # Verbose output
  python run_all.py --list-steps       # List available steps

Steps:
  1. Generate hypergraph structure
  2. Compute node measures (μ)
  3. Compute hyperedge measures (ν)
  4. Compute hypernetwork function (ω)
  5. Run HyperCOT optimization
  6. Generate visualizations
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file (default: config.yaml)"
    )

    parser.add_argument(
        "--steps", "-s",
        type=str,
        default=None,
        help="Comma-separated list of step numbers to run (e.g., '1,2,3')"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List available pipeline steps and exit"
    )

    args = parser.parse_args()

    # List steps and exit
    if args.list_steps:
        print("\nAvailable pipeline steps:")
        print("-" * 50)
        for num, (name, _) in PIPELINE_STEPS.items():
            print(f"  {num}. {name}")
        print()
        return 0

    # Parse steps
    steps = None
    if args.steps:
        try:
            steps = [int(s.strip()) for s in args.steps.split(",")]
        except ValueError:
            print(f"Error: Invalid step format '{args.steps}'")
            print("Use comma-separated numbers, e.g., '1,2,3'")
            return 1

    # Run pipeline
    success = run_pipeline(
        steps=steps,
        config_path=args.config,
        verbose=args.verbose
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
