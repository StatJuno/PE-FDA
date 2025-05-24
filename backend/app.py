import streamlit as st
from pages.first_page import first_page
from pages.second_page import second_page

def main():
    st.set_page_config(
        page_title="Pose Estimation with Functional Data Analysis",
        page_icon="💪",
        layout="wide",  # 전체 화면 너비로 설정
    )

    # Add this function to your script to include the CSS
    def add_custom_css():
        css = """
        <style>
            /* Center the title */
            .centered-title {
                text-align: center;
                margin-top: 0;
                margin-bottom: 1rem;
                font-size: 3rem;
                font-weight: bold;
                color: #ff4b2b;
                text-shadow: 2px 2px 4px #000000;
            }
            /* General page styling */
            .main {
                background-color: #f5f5f5;
                font-family: 'Arial', sans-serif;
                text-align: left;
            }

            /* Header styling */
            h1, h2, h3 {
                color: #333333;
            }

            h1 {
                margin-bottom: 1rem;
                font-size: 2.5rem;
            }

            h2 {
                margin-top: 1rem;
                font-size: 2rem;
            }

            /* Buttons */
            .stButton>button {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 1rem;
                transition: background-color 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #0056b3;
                color: #ffffff;
            }

            /* Selectbox */
            .stSelectbox div[data-baseweb="select"] {
                background-color: #f9f9f9;
                border-radius: 5px;
            }

            /* File uploader styling */
            .stFileUploader {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 1rem;
                margin-top: 1rem;
            }

            /* Video player */
            .stVideo {
                margin: 20px auto;
                display: block;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            /* Divider */
            hr {
                border: none;
                border-top: 1px solid #cccccc;
                margin: 2rem 0;
            }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    # Call this function at the beginning of your app
    add_custom_css()

    st.markdown(
        """
        <h1 style="text-align: center; font-size: 100px; margin-top: 20px; font-family: Roboto;">
            Pose Estimation with Functional Data Analysis
        </h1>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = 1

    # Page navigation
    if st.session_state.page == 1:
        first_page()
    elif st.session_state.page == 2:
        second_page()

if __name__ == "__main__":
    main()
