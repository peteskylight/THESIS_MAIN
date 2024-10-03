import pickle
import os

def read_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print("Contents of the pickle file:")
            print(data)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_path = os.path.abspath('RENOVATION/cache/inference_temp_results.pkl')
read_pickle_file(file_path)
