import argparse
import encode_data

parser = ArgumentParser(description="Trains a neural network to identify kozac consensus sequence")
parser.add_argument("--encoder", type=str, default="binary", help="How you want to encode data (one_hot, binary, frequency)")

