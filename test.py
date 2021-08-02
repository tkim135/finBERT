import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--start_seed', type=int, required=True)
parser.add_argument('--end_seed', type=int, required=True)
parser.add_argument('--type', type=str, required=True)

args = parser.parse_args()

lm_path = args.model_name_or_path
cl_path = "/scratch/varunt/finbert_clpath/" + args.type + "/" + "seed_" + str(42) + "/" + lm_path + "/"
print(cl_path)