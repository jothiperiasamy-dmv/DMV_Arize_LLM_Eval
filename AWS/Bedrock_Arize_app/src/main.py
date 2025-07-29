from io import BytesIO
import streamlit as st
from streamlit_chat import message
from streamlit_feedback import streamlit_feedback
from src.model import get_aws_agent_model_response
import os
import pandas as pd


def agent():
    col1, col2, col3, col4 = st.columns((1, 3, 4, 1.5))
    col11, col22, col33, col44 = st.columns((1, 3, 4, 1.5))
    m1, m2, m3 = st.columns([1, 10, 1])
    with col2:
        st.write("###### ")
        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Select Model</span></p>", unsafe_allow_html=True)
    with col3:
        # st.write(" ")
        vAR_model = st.selectbox("",["Select","AWS Agents(Claude LLM)", "Openai Models"], index=0, key="model_select")
    if vAR_model != "Select":
        with m2:
            st.write('### ')
            ########################################### chatbot UI ###############################################
            if 'history' not in st.session_state:
                    st.session_state['history'] = []

            if 'generated' not in st.session_state:
                st.session_state['generated'] = ["Greetings! I am LLMAI Live Agent. How can I help you?"]

            if 'past' not in st.session_state:
                st.session_state['past'] = ["We are delighted to have you here in the LLMAI Live Agent Chat room!"]
            
            #container for the chat history
            response_container = st.container()
            
            #container for the user's text input
            container = st.container()
            with container:
                with st.form(key='my_form', clear_on_submit=True):
                    user_input = st.text_input("Prompt:", placeholder="How can I help you?", key='input')
                    submit_button = st.form_submit_button(label='Interact with LLM')
                    
                if submit_button and user_input:
                    # messages_history.append(HumanMessage(content=user_input))
                    
                    vAR_response, df = get_aws_agent_model_response(user_input)
                    st.session_state.user_input = user_input  
                    st.dataframe(df, use_container_width=True)
                    # Convert DataFrame to Excel in memory
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='Sheet1')
                    processed_data = output.getvalue()
                    # Excel download button
                    st.download_button(
                        label="Download Report as Excel",
                        data=processed_data,
                        file_name="model.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )               
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(vAR_response)

            if st.session_state['generated']:
                    with response_container:
                        for i in range(len(st.session_state['generated'])):
                            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                            message(st.session_state["generated"][i], key=str(i+55), avatar_style="thumbs")
                            if i != 0:
                                feedback = streamlit_feedback(
                                align="flex-start",
                                feedback_type="thumbs",
                                optional_text_label="[ Human Feedback Optional ] Please provide an explanation",
                                key=f"thumbs_{i}"
                                )