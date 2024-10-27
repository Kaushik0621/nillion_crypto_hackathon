
import aivm_client as aic 

MODEL_NAME = "enhanced_model" 
aic.upload_lenet5_model("./lenet5_model.pth", MODEL_NAME)