#!/usr/bin/env python3
"""
KAN Lottery Ticket Project – Main Entry Point
==============================================

Usage:
    python main.py --quick          # smoke test (2 seeds, fast)
    python main.py                  # full run (5 seeds, all tasks)
    python main.py --plot-only      # regenerate figures from existing results
    python main.py --verbose        # verbose training output
"""
import os, sys, argparse

sys.path.insert(0, os.path.dirname(__file__))

def main():
    parser = argparse.ArgumentParser(description="KAN Lottery Ticket Experiments")
    parser.add_argument("--quick",     action="store_true",
                        help="Smoke-test: 2 seeds, reduced epochs")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print per-epoch training info")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training; regenerate plots from existing results")
    args = parser.parse_args()

    if not args.plot_only:
        from experiment_runner import run_all
        print("=" * 60)
        print("  KAN Lottery Ticket – Experiment Runner")
        print(f"  mode: {'quick' if args.quick else 'full'}")
        print("=" * 60)
        run_all(verbose=args.verbose, quick=args.quick)

    print("\nGenerating figures …")
    from plot import (load_results, plot_accuracy_vs_sparsity,
                      plot_jaccard, plot_resilience,
                      plot_kan_complexity, print_summary)
    results = load_results()
    plot_accuracy_vs_sparsity(results)
    plot_jaccard(results)
    plot_resilience(results)
    plot_kan_complexity(results)
    print_summary(results)
    print("\nDone. Results and figures in kan_lottery/results/")


if __name__ == "__main__":
    main()
