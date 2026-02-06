#!/usr/bin/env python3
import argparse
from src.pipeline import train_pipeline


def main():
    p = argparse.ArgumentParser(description="Train NeRF")
    p.add_argument("--basedir", type=str, default="data/lego",
                   help="Path to dataset folder containing transforms_train.json and images")
    p.add_argument("--N_iters", type=int, default=200_000, help="Training iterations")
    p.add_argument("--N_rand", type=int, default=1024, help="Random rays per batch")
    p.add_argument("--N_samples", type=int, default=64, help="Samples per ray")
    p.add_argument("--lrate", type=float, default=5e-4, help="Learning rate")
    p.add_argument("--near", type=float, default=2.0, help="Near plane")
    p.add_argument("--far", type=float, default=6.0, help="Far plane")
    p.add_argument("--multires", type=int, default=10, help="Position encoding frequencies")
    p.add_argument("--multires_views", type=int, default=4, help="View encoding frequencies")
    args = p.parse_args()
    train_pipeline(args)


if __name__ == "__main__":
    main()
