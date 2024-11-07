import streamlit as st
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import os
import warnings
from PIL import Image  # Added import for PIL

# Suppress warnings
warnings.filterwarnings('ignore')

# Set current directory and load the model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "you_look_like.h5")
model = load_model(model_path)

# List of class labels (Bollywood actors)
class_labels = ['abhay_deol', 'adil_hussain', 'ajay_devgn', 'akshay_kumar', 'akshaye_khanna', 'amitabh_bachchan',
                'amjad_khan', 'amol_palekar', 'amole_gupte', 'amrish_puri', 'anil_kapoor', 'annu_kapoor', 
                'anupam_kher', 'anushka_shetty', 'arshad_warsi', 'aruna_irani', 'ashish_vidyarthi', 'asrani', 
                'atul_kulkarni', 'ayushmann_khurrana', 'boman_irani', 'cat', 'chiranjeevi', 'chunky_panday', 
                'danny_denzongpa', 'darsheel_safary', 'deepika_padukone', 'deepti_naval', 'dev_anand', 
                'dharmendra', 'dilip_kumar', 'dimple_kapadia', 'dog', 'farhan_akhtar', 'farida_jalal', 
                'farooq_shaikh', 'girish_karnad', 'govinda', 'gulshan_grover', 'hrithik_roshan', 'huma_qureshi', 
                'irrfan_khan', 'jaspal_bhatti', 'jeetendra', 'jimmy_sheirgill', 'johnny_lever', 'kader_khan', 
                'kajol', 'kalki_koechlin', 'kamal_haasan', 'kangana_ranaut', 'kay_kay_menon', 'konkona_sen_sharma', 
                'kulbhushan_kharbanda', 'lara_dutta', 'madhavan', 'madhuri_dixit', 'mammootty', 'manoj_bajpayee', 
                'manoj_pahwa', 'mehmood', 'mita_vashisht', 'mithun_chakraborty', 'mohanlal', 'mohnish_bahl', 
                'mukesh_khanna', 'mukul_dev', 'nagarjuna_akkineni', 'nana_patekar', 'nandita_das', 'nargis', 
                'naseeruddin_shah', 'navin_nischol', 'nawazuddin_siddiqui', 'neeraj_kabi', 'nirupa_roy', 'om_puri', 
                'pankaj_kapur', 'pankaj_tripathi', 'paresh_rawal', 'pawan_malhotra', 'pooja_bhatt', 'prabhas', 
                'prabhu_deva', 'prakash_raj', 'pran', 'prem_chopra', 'priyanka_chopra', 'raaj_kumar', 'radhika_apte', 
                'rahul_bose', 'raj_babbar', 'raj_kapoor', 'rajat_kapoor', 'rajesh_khanna', 'rajinikanth', 
                'rajit_kapoor', 'rajkummar_rao', 'rajpal_yadav', 'rakhee_gulzar', 'ramya_krishnan', 'ranbir_kapoor', 
                'randeep_hooda', 'rani_mukerji', 'ranveer_singh', 'ranvir_shorey', 'ratna_pathak_shah', 'rekha', 
                'richa_chadha', 'rishi_kapoor', 'riteish_deshmukh', 'sachin_khedekar', 'saeed_jaffrey', 
                'saif_ali_khan', 'salman_khan', 'sanjay_dutt', 'sanjay_mishra', 'shabana_azmi', 'shah_rukh_khan', 
                'sharman_joshi', 'sharmila_tagore', 'shashi_kapoor', 'shreyas_talpade', 'smita_patil', 
                'soumitra_chatterjee', 'sridevi', 'sunil_shetty', 'sunny_deol', 'tabu', 'tinnu_anand', 'utpal_dutt', 
                'varun_dhawan', 'vidya_balan', 'vinod_khanna', 'waheeda_rehman', 'zarina_wahab', 'zeenat_aman']

# Streamlit UI
st.title("Bollywood Actor Image Classifier")
st.write("Upload an image to predict which Bollywood actor it looks like!")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open the uploaded image using PIL
        image = load_img(uploaded_file, target_size=(200, 200))  # resize the image
        img_array = img_to_array(image)  # Convert to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize image
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make predictions
        predictions = model.predict(img_array)

        # Get the predicted class
        predicted_class_index = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class_index] if predicted_class_index < len(class_labels) else "Unknown"

        # Display results
        st.write(f"Predicted Actor: **{predicted_label}** with confidence: {predictions[0][predicted_class_index]*100:.2f}%")

        # Show top N predictions if probability is less than 85%
        max_prob = max(predictions[0])
        top_n = 1 if max_prob >= 0.85 else 2  # Adjust top_n based on max probability
        top_n_indices = np.argsort(predictions[0])[-top_n:]

        st.write(f"Top {top_n} predictions:")
        for idx in reversed(top_n_indices):  # Reverse to show highest probability first
            if idx < len(class_labels):
                st.write(f"Class: **{class_labels[idx]}**, Probability: {predictions[0][idx]*100:.2f}%")
            else:
                st.write(f"Class: **Unknown**, Probability: {predictions[0][idx]*100:.2f}%")

    except Exception as e:
        st.error(f"Error processing the image: {e}")
