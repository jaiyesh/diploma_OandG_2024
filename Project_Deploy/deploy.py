import pandas as pd
import gradio as gr
from pycaret.regression import load_model, predict_model

# Load the trained model
rf = load_model("rf_volvep12")

# Define feature names
features_name = ['onstreaminject_HRS',
                'BORE_WI_VOL', 'ON_STREAM_HRS',
                'AVG_DP_TUBING',
                'AVG_ANNULUS_PRESS',
                'AVG_CHOKE_SIZE_P in percentage',
                'AVG_WHP_P', 'AVG_WHT_P',
                'DP_CHOKE_SIZE']

# Define a prediction function
def predict(*features):
    # Convert the input tuple into a DataFrame
    input_data = pd.DataFrame([features], columns=features_name)
    prediction = predict_model(rf, data=input_data)
    print(prediction)
    return prediction['prediction_label'][0]

# Dynamically generate Gradio inputs
inputs = [gr.Number(label=feature) for feature in features_name]

# Define the Gradio interface
outputs = gr.Textbox(label="Oil Production Estimation")
interface = gr.Interface(fn=predict, inputs=inputs, outputs=outputs)

# Launch the interface
interface.launch(share=True)
