from src.data import load_dataset, scale_dataset,split_dataset
from src.model.model import build_model, compile_model
from src.models.callbacks import get_callbacks

def main():
	orig_ds = load_dataset("data", ["no_cloud", "cloud"], 
							class_names = ["no_cloud", "cloud"],
							batch_size=10,
							image_size=(256,256),
							shuffle=True)
	scaled_ds = scale_dataset(orig_ds)
	train_ds, val_ds, test_ds = split_dataset(scaled_ds)

	model = build_model(
		input_shape=(256, 256, 3),
		conv_filters=[16, 32],
		kernel_sizes=[(5, 5), (3, 3)],
		conv_activations=['relu', 'relu'],
		dense_activation='sigmoid'
	)

	model = compile_model(model, optimizer='adam', loss='binary_crossentropy')
      
	callbacks = get_callbacks(log_dir='logs', model_path='models/best_model_cloud_nocloud.keras')
    
	history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks)

		


if __name__ == "__main__":
    main()