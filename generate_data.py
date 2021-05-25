from read_dataset_for_constraint import generate_dataset, switch_dataset



if __name__ == "__main__":
	datasets = ["wine", "bank_marketing", "magic_gamma", "pima_indian_diabetes", "ionosphere", "german_credit", "waveform"]

	for dataset in datasets:
		for i in range(10,15):
			#switch_dataset("qsar_oral_toxicity")()
			generate_dataset(dataset, number=i)
