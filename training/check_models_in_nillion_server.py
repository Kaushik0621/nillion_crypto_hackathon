import aivm_client as aic

# List all supported models
available_models = aic.get_supported_models()
print(available_models)