from process_emg import OnlineEMGPredictor

# Initialize the online predictor
predictor = OnlineEMGPredictor(config_filename='pipeline_mlp_to_cnn.yaml')

# Path to single EMG file
emg_file = "F:/SMT_MASTERPROJEKT/biosignal_toolbox/data/jte/emg/BU62D/test500.txt"  # Replace with actual path 24072025_BU62D_0g_complex_1

# Run the complete pipeline
try:
    predictions = predictor.run(emg_file)

    # Display results
    print("=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"Shape: {predictions.shape}")

except Exception as e:
    print(f"Error during prediction: {e}")
    raise