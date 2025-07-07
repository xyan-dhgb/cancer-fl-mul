def main():
    """
    Main function to execute the complete workflow
    """
    print("=== Breast Cancer Detection System with Multimodal Learning ===")
    
    # Set file paths - replace with your actual paths
    csv_path = "data/mias_derived_info.csv"
    image_dir = "data/MIAS"
    
    # 1. Load and explore data
    try:
        df = load_csv_data(csv_path)
        explore_data(df)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found.")
        print("Using a small sample dataframe for demonstration purposes.")
        # Create a small sample dataframe for demonstration
        sample_data = {
            'REFNUM': ['mdb001', 'mdb002', 'mdb003'],
            'BG': ['G', 'F', 'G'],
            'CLASS': ['NORM', 'CIRC', 'SPIC'],
            'X': [None, 540, 470],
            'Y': [None, 565, 480],
            'RADIUS': [None, 60, 45],
            'DENSITY': ['B', 'A', 'B'],
            'BI-RADS': ['BI-RADS 1', 'BI-RADS 3', 'BI-RADS 4'],
            'SEVERITY': ['Normal', 'Benign', 'Malignant']
        }
        df = pd.DataFrame(sample_data)
        print("Created sample dataframe:")
        print(df)
    
    # 2. Preprocess data
    processed_df = preprocess_csv_data(df)
    print("\nPreprocessed dataframe:")
    print(processed_df.head())
    
    # 3. Prepare multimodal data
    print("\nPreparing multimodal data...")
    try:
        X_tab_train, X_tab_test, X_img_train, X_img_test, y_train, y_test = prepare_multimodal_data(
            processed_df, image_dir
        )
        print(f"Training data shapes: Tabular: {X_tab_train.shape}, Images: {X_img_train.shape}")
    except:
        print("Warning: Could not prepare actual image data. Using dummy data for demonstration.")
        # Create dummy data for demonstration
        X_tab_train = np.random.randn(80, 7)
        X_tab_test = np.random.randn(20, 7)
        X_img_train = np.zeros((80, 224, 224, 1))
        X_img_test = np.zeros((20, 224, 224, 1))
        y_train = to_categorical(np.random.randint(0, 3, 80))
        y_test = to_categorical(np.random.randint(0, 3, 20))
    
    # 4. Build model
    print("\nBuilding multimodal model...")
    model = build_multimodal_model(
        tabular_shape=X_tab_train.shape[1],
        image_shape=X_img_train.shape[1:],
        num_classes=3
    )
    model.summary()
    
    # 5. Train model
    print("\nTraining model...")
    trained_model, history = train_model(
        model,
        X_tab_train, X_img_train, y_train,
        X_tab_test, X_img_test, y_test,
        epochs=5  # Reduced for demonstration
    )
    
    # 6. Evaluate model
    print("\nEvaluating model...")
    y_pred, y_pred_prob = evaluate_model(trained_model, X_tab_test, X_img_test, y_test)
    
    # 7. Feature importance analysis
    print("\nAnalyzing feature importance...")
    analyze_feature_importance(trained_model, X_tab_test, y_test)
    
    # 8. Sample prediction
    print("\nMaking a sample prediction...")
    new_patient = {
        'REFNUM': 'new_patient_001',
        'BG': 'G',
        'CLASS': 'CIRC',
        'X': 520,
        'Y': 380,
        'RADIUS': 45,
        'DENSITY': 'B',
        'BI-RADS': 'BI-RADS 3',
        'SEVERITY': 'Benign'  # May need to adjust this value based on expected input
    }
    
    # Sample image path for the new patient
    sample_image_path = os.path.join(image_dir, "sample.pgm")
    if not os.path.exists(sample_image_path):
        print(f"Warning: Image file '{sample_image_path}' not found. Using zero array.")
        # Create a dummy image file for demonstration
        dummy_img = np.zeros((224, 224))
        cv2.imwrite("dummy_sample.pgm", dummy_img)
        sample_image_path = "dummy_sample.pgm"
        
    # Make prediction
    diagnosis_result = predict_cancer_diagnosis(trained_model, new_patient, sample_image_path)
    print(f"Diagnosis: {diagnosis_result['diagnosis']}")
    print(f"Confidence: {diagnosis_result['confidence']:.2f}%")
    print("Probabilities:")
    for class_name, prob in diagnosis_result['probabilities'].items():
        print(f"  {class_name}: {prob:.2f}%")
    
    # 9. Visualize the result
    print("\nVisualizing diagnosis result...")
    try:
        visualize_diagnosis(sample_image_path, diagnosis_result)
        
        # Create a sample heatmap
        heatmap = np.zeros((512, 512))
        heatmap[200:300, 200:300] = np.linspace(0, 1, 100)[:, np.newaxis] * np.linspace(0, 1, 100)[np.newaxis, :]
        visualize_diagnosis(sample_image_path, diagnosis_result, heatmap)
    except Exception as e:
        print(f"Error visualizing diagnosis: {e}")
    
    print("\nBreast Cancer Detection workflow completed successfully!")

if __name__ == "__main__":
    main()