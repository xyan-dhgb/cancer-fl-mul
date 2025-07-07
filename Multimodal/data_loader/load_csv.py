def load_csv_data(file_path):
    """
    Load the CSV data and display basic information
    """
    print("Loading CSV data...")
    df = pd.read_csv(file_path)
    return df