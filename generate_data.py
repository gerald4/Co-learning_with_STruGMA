from read_dataset_for_constraint import generate_dataset



if __name__ == "__main__":

	for i in range(5):
		generate_dataset("student_performance", number=i)