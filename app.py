import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io
from streamlit_option_menu import option_menu
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="Sortiva AI", page_icon="random", layout="centered")

hide_streamlit_style = """ 
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Home", "Stats","Learn"],
    icons=["house", "sun", "lightbulb"],
    orientation="horizontal",
)

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()

if selected == "Home":

    # LOGO.
    logo = current_dir / "assets" / "logo.jpeg"
    logo = Image.open(logo)

    st.image(logo, width=500)

    # Load your trained model (adjust the path if needed)
    @st.cache_resource(show_spinner=False)
    def load_model():
        model = tf.keras.models.load_model('best_model.keras')
        return model

    model = load_model()

    # Define the target image size (same as used during training)
    IMG_HEIGHT, IMG_WIDTH = 224, 224

    # Define your class mapping (update if necessary)
    class_indices = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}
    idx2label = {v: k for k, v in class_indices.items()}

    def load_and_preprocess_image(image_data):
        """Load an image from file bytes and preprocess it."""
        # Open the image using PIL
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        # Resize the image
        img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
        # Convert image to numpy array
        img_array = image.img_to_array(img_resized)
        # Expand dimensions to match model input (1, IMG_HEIGHT, IMG_WIDTH, 3)
        img_array = np.expand_dims(img_array, axis=0)
        # Preprocess the image using MobileNetV2's preprocessing
        img_array = preprocess_input(img_array)
        return img_array, img

    st.subheader("Zero Trash with AI", divider=True)
    st.success("Take ur trash a photo and then 👇👇")
    col1, col2 = st.columns([1, 2], border=True)
    with col1:
        # File uploader for image input
        uploaded_file = st.file_uploader(label="", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read the image file
            image_data = uploaded_file.read()
            # Preprocess the image and get original image for display
            img_array, original_img = load_and_preprocess_image(image_data)

            # Make prediction with the model
            predictions = model.predict(img_array)
            predicted_class = int(np.argmax(predictions, axis=1)[0])
            predicted_label = idx2label[predicted_class]
            confidence = predictions[0][predicted_class]

            with col2:
                # Display results in Streamlit with a reduced image width (e.g., 300 pixels)
                st.image(original_img, caption="Preview", width=300)
                st.info(f"**Prediction:** {predicted_label}")
                st.info(f"**Confidence:** {confidence * 100:.2f}%")

                # Expanded recycling suggestions with more detailed information
                recycling_suggestions = {
                    'cardboard': (
                        "Recycle cardboard by flattening the boxes and keeping them dry. "
                        "Place them in the designated cardboard recycling bin. "
                        "Ensure there is no food residue attached."
                    ),
                    'glass': (
                        "Glass should be cleaned and sorted by color (if required by your local guidelines). "
                        "Place it in the glass recycling container. "
                        "Avoid mixing with ceramics or mirrors."
                    ),
                    'metal': (
                        "Metals such as aluminum and steel should be rinsed and sorted. "
                        "Recycle them in the metal recycling bin. "
                        "Scrap metal collectors may offer better returns for large quantities."
                    ),
                    'paper': (
                        "Recycle paper by ensuring it is clean and dry. "
                        "Flatten and bundle paper items before placing them in the paper recycling bin. "
                        "Avoid mixing with contaminated or greasy paper products."
                    ),
                    'plastic': (
                        "Plastics should be rinsed to remove food residues and then sorted by type if possible. "
                        "Check for the recycling symbol on the plastic. "
                        "Place in the appropriate plastic recycling bin."
                    ),
                    'trash': (
                        "If an item cannot be recycled, dispose of it as general waste. "
                        "Consider ways to reduce waste or reuse items before discarding. "
                        "Consult your local waste management guidelines for hazardous or electronic waste."
                    )
                }

                suggestion = recycling_suggestions.get(predicted_label, "No suggestion available.")
                st.success(f"**Recycling Suggestion:** {suggestion}")

elif selected == ("Stats"):

    st.subheader("Stats don't lie", divider=True)
    fig = px.bar(
        x=["Cardboard", "Glass", "Paper", "Metal", "Trash", "Plastic"],
        y=[120, 90, 100, 50, 40, 100],
        title="📊 Weekly Waste Disposed in grams",
        color=[120, 118, 119, 117, 121, 115],
        color_continuous_scale='reds'
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.area(
        x=["Wk 1", "Wk 2", "Wk 3", "Wk 4", "Wk 5"],
        y=[5, 6, 7, 6, 8],
        title="😊 Proper Waste Disposal Rewards",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

elif selected == "Learn":
    st.subheader("Be waste-SMART", divider=True)

    article_101 = st.expander(label="Waste Less, Live More.", expanded=True)
    article_101.write("Discover smart strategies to reduce your footprint. Master recycling essentials and composting hacks to build a cleaner, sustainable future today.")

    article_102 = st.expander(label="Transform Your Trash.", expanded=True)
    article_102.write("Revolutionize your waste habits with expert insights. From zero-waste living to efficient sorting, learn how to impact the planet positively.")

    videos = st.expander(label="Video Tutorials", expanded=True)
    videos.video("https://www.youtube.com/watch?v=HgEo7YnvJs0&pp=ygUad2FzdGUgbWFuYWdlbWVudCB0dXRvcmlhbHM%3D")

    videos.video("https://www.youtube.com/watch?v=Qyu-fZ8BOnI&pp=ygUad2FzdGUgbWFuYWdlbWVudCB0dXRvcmlhbHM%3D")
