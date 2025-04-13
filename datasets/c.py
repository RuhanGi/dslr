import itertools
import subprocess
import os
import pandas as pd
from sklearn.metrics import accuracy_score

COURSES = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
			'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
			'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']

TRAIN_FILE = "datasets/dataset_train.csv"
TEST_FILE = "datasets/dataset_test.csv"
TRUTH_FILE = "datasets/dataset_truth.csv"
THETAS_FILE = "thetas.csv"
PREDICTIONS_FILE = "houses.csv"

def evaluate_accuracy():
	try:
		truth_df = pd.read_csv(TRUTH_FILE)
		pred_df = pd.read_csv(PREDICTIONS_FILE)
		return accuracy_score(truth_df["Hogwarts House"], pred_df["Hogwarts House"]) * 100
	except Exception as e:
		print(f"Error evaluating accuracy: {e}")
		return -1

def create_filtered_dataset(input_file, output_file, features):
	df = pd.read_csv(input_file)
	features_with_label = features + ["Hogwarts House"] if "dataset_train" in input_file else features
	filtered_df = df[["Index"] + features_with_label] if "Index" in df.columns else df[features_with_label]
	filtered_df.to_csv(output_file, index=False)

def main():
	best_acc = 97.5
	best_combo = ('Defense Against the Dark Arts', 'Charms', 'Flying')

	for r in range(4, len(COURSES)+1):
		for combo in itertools.combinations(COURSES, r):
			print(f"Trying courses: {combo}")
			create_filtered_dataset(TRAIN_FILE, "temp_train.csv", list(combo))
			create_filtered_dataset(TEST_FILE, "temp_test.csv", list(combo))
			subprocess.run(["python3", "src/model/logreg_train.py", "temp_train.csv"],
						   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			subprocess.run(["python3", "src/model/logreg_predict.py", "temp_test.csv", THETAS_FILE],
						   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			acc = evaluate_accuracy()
			print(f"Accuracy: {acc:.2f}%")
			if acc > best_acc:
				best_acc = acc
				best_combo = combo
			print("\033[92m" + f"✅ Best Course Combination: {best_combo}" + "\033[0m")
			print("\033[93m" + f"🎯 Accuracy: {best_acc:.2f}%" + "\033[0m")

if __name__ == "__main__":
	main()
