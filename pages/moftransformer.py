import os
import re
import io
import json
from typing import Dict
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import streamlit as st


def get_models() -> Dict[str, str]:
    # Models
    df = pd.read_excel('assets/models.xlsx')
    df = df.fillna('None')

    models = []
    for line in df.iloc():
        model = {i: v for i, v in line.items()}
        models.append(model)
    return models


def str_to_html(text: str) -> str:
    text = re.sub(r"\^(?P<sup>\d+)",
                  lambda t: f"<sup>{t.group('sup')}</sup>", text)
    text = re.sub(r"\_(?P<sub>\d+)",
                  lambda t: f"<sub>{t.group('sub')}</sub>", text)
    return text


def create_model_box(model: dict):
    name = str_to_html(model["name"])
    text = f'<div class="box"><h3>{name}</h3>'

    unit = str_to_html(model['unit'])
    text += f"<p>Unit: {unit}</p>"
    condition = str_to_html(model['condition'])
    text += f"<p>Condition: {condition}</p>"
    # text += f"<p>Upload Date: {model['date']}</p>"
    text += f"<p>Upload : {model['uploader']} ({model['date']})</p>"
    # keys = ['unit', 'condition']
    # for key, value in model.items():
    #    if key == 'name':
    #        continue
    #    text += f"<p>{key.capitalize()}: {value}</p>"
    text += '</div>'

    st.markdown(
        text,
        # f'<div class="box">'
        # f'<h3>{model["name"]}</h3>'
        # f'<p>'
        # f'Unit: {model["unit"]}</p><p>'
        # f'Condition: {model["condition"]}'
        # '</p>'
        # f'</div>',
        unsafe_allow_html=True
    )


def send_email(subject, files, info_dict):
    sender_email = "dudgns4675@gmail.com"
    receiver_email = "dudgns1675@kaist.ac.kr"
    password = st.secrets["email_passwd"]

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    for file in files:
        part = MIMEBase('application', "octet-stream")
        part.set_payload(file.getvalue())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        'attachment; filename="%s"' % file.name)
        msg.attach(part)

    # Use your SMTP server details
    info_str = json.dumps(info_dict, indent=4)
    info_file = io.StringIO(info_str)
    part = MIMEBase('application', "octet-stream")
    part.set_payload(info_file.getvalue())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="info.json"')
    msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, password)
    server.send_message(msg)
    server.quit()


# Custom CSS to style the boxes
st.markdown("""
<style>
.box {
    border: 2px solid #EAEAEA;
    padding: 10px;
    border-radius: 10px;
    margin: 10px 0;
    background-color: #FAFAFA;
    font-size: 4px;
}
.box h3 {
    font-size: 20px; /* Adjust font size as needed */
    color: #333333; /* Optional: Change the font color */
    margin: 10; /* Optional: Remove default margins */
}
.box p {
    font-size: 15px; /* Smaller font size for description */
    color: #AAAAAA;
    margin: 0; /* Remove default margins */
}
            
.box:hover {
    background-color: #000000;
}
</style>
""", unsafe_allow_html=True)

# st.set_page_config(layout="wide")
st.title('🚀 MOFTransformer models')
st.subheader('Fine-tuned model')

# Creating a session state to store the selected model
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

num_columns = 2
models = get_models()
rows = [models[i: i+num_columns] for i in range(0, len(models), num_columns)]

# Displaying boxes for each model
for row in rows:
    cols = st.columns(num_columns)
    for idx, col in enumerate(cols):
        if len(row) <= idx:
            continue
        model = row[idx]
        with col:
            create_model_box(model)

st.subheader('\n\n')
st.subheader('Upload model')

st.write(
    """To successfully upload your model files for integration with ChatMOF, please ensure you include the following two key files that are generated during the training of MOFTransformer:

1. **Fine-Tuned Model Weight (model.ckpt)**: This file contains the fine-tuned weights of your model, which are crucial for accurate performance and predictions.

2. **Model Configuration File (hparams.yaml)**: This file outlines the configuration parameters of your model, providing essential context and settings used during training.

In addition to the files, we also require specific information about your model to ensure seamless integration and functionality within ChatMOF:

1. **Model name**: Provide the name of your model. This will be used as the primary identifier within ChatMOF.
2. **Description**: A brief yet comprehensive description of your model. This should highlight its unique features and capabilities.
3. **Unit**: Specify the unit of property that was used during the learning process.
4. **Condition**: Detail the condition(s) under which the property was learned. This helps in understanding the applicability and limitations of the model.
5. **Others**: Any additional information that might be relevant, such as whether log values were utilized, or any specific methodologies employed.
6. **Uploader**: Your name or the name of the person uploading the model. This is important for attribution and reference.

Please ensure all information is accurate and complete. Incomplete or incorrect submissions may lead to delays in integration with ChatMOF.
"""
)

model_file = st.file_uploader(
    "Choose a model.ckpt file", accept_multiple_files=False, type=['ckpt'])

yaml_file = st.file_uploader(
    "Choose a hparams.yaml file", accept_multiple_files=False, type=['yaml'])

name = st.text_input('Model name')
description = st.text_input('Description')
unit = st.text_input('Unit')
condition = st.text_input('Condition')
others = st.text_input('Others')
uploader = st.text_input('Uploader')

info_dict = {
    'name': name,
    'description': description,
    'unit': unit,
    'condition': condition,
    'others': others,
    'uploader': uploader,
}

send_button = st.button("Upload")

if send_button:
    if model_file and yaml_file and uploader:
        send_email("New Model Files", [model_file, yaml_file], info_dict)
        st.success("Files sent successfully!")
    else:
        st.warning('You must upload model, config, and information')
